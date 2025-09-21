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
    ):
        # pd_op.conv2d: (2x256x96x96xf32) <- (2x256x96x96xf32, 256x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_13, parameter_158, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_13, parameter_158

        # pd_op.batch_norm_: (2x256x96x96xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x96x96xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_0,
                parameter_157,
                parameter_156,
                parameter_155,
                parameter_154,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_154, parameter_155, parameter_156, parameter_157

        # pd_op.conv2d: (2x256x48x48xf32) <- (2x256x48x48xf32, 256x256x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_14, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_14, parameter_153

        # pd_op.batch_norm_: (2x256x48x48xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x48x48xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_1,
                parameter_152,
                parameter_151,
                parameter_150,
                parameter_149,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_149, parameter_150, parameter_151, parameter_152

        # pd_op.conv2d: (2x256x24x24xf32) <- (2x256x24x24xf32, 256x256x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_15, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_15, parameter_148

        # pd_op.batch_norm_: (2x256x24x24xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x24x24xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_144, parameter_145, parameter_146, parameter_147

        # pd_op.flatten: (2x256x9216xf32) <- (2x256x96x96xf32)
        flatten_0 = paddle._C_ops.flatten(batch_norm__0, 2, 3)

        # pd_op.transpose: (2x9216x256xf32) <- (2x256x9216xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.flatten: (2x256x2304xf32) <- (2x256x48x48xf32)
        flatten_1 = paddle._C_ops.flatten(batch_norm__6, 2, 3)

        # pd_op.transpose: (2x2304x256xf32) <- (2x256x2304xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_1, [0, 2, 1])
        del flatten_1

        # pd_op.flatten: (2x256x576xf32) <- (2x256x24x24xf32)
        flatten_2 = paddle._C_ops.flatten(batch_norm__12, 2, 3)

        # pd_op.transpose: (2x576x256xf32) <- (2x256x576xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # builtin.combine: ([2x9216x256xf32, 2x2304x256xf32, 2x576x256xf32]) <- (2x9216x256xf32, 2x2304x256xf32, 2x576x256xf32)
        combine_0 = [transpose_0, transpose_1, transpose_2]

        # pd_op.concat: (2x12096x256xf32) <- ([2x9216x256xf32, 2x2304x256xf32, 2x576x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.full: (2x1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [2, 1], float("2"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (2x1x4xf32) <- ()
        full_2 = paddle._C_ops.full(
            [2, 1, 4],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (2x1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [2, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_0

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(data_17, full_int_array_0)
        del data_17

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_2

        # pd_op.set_value_with_tensor_: (2x1xi32) <- (2x1xi32, 1xi32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            full_1,
            squeeze_0,
            full_int_array_1,
            full_int_array_2,
            full_int_array_2,
            [0],
            [0],
            [],
        )
        del full_1, squeeze_0

        # pd_op.set_value_with_tensor_: (2x1x4xf32) <- (2x1x4xf32, 1x4xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(
            full_2,
            data_19,
            full_int_array_1,
            full_int_array_2,
            full_int_array_2,
            [0],
            [0],
            [],
        )
        del data_19, full_2

        # pd_op.set_value_: (2x1xf32) <- (2x1xf32, 1xi64, 1xi64, 1xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_3,
            full_int_array_1,
            full_int_array_2,
            full_int_array_2,
            [0],
            [0],
            [],
            [1],
            [float("1")],
        )
        del full_3

        # pd_op.squeeze: (1xi32) <- (1x1xi32, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(data_18, full_int_array_0)
        del data_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_3

        # pd_op.set_value_with_tensor_: (2x1xi32) <- (2x1xi32, 1xi32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__0,
            squeeze_1,
            full_int_array_2,
            full_int_array_3,
            full_int_array_2,
            [0],
            [0],
            [],
        )
        del set_value_with_tensor__0, squeeze_1

        # pd_op.set_value_with_tensor_: (2x1x4xf32) <- (2x1x4xf32, 1x4xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(
            set_value_with_tensor__1,
            data_20,
            full_int_array_2,
            full_int_array_3,
            full_int_array_2,
            [0],
            [0],
            [],
        )
        del data_20, set_value_with_tensor__1

        # pd_op.set_value_: (2x1xf32) <- (2x1xf32, 1xi64, 1xi64, 1xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_2,
            full_int_array_3,
            full_int_array_2,
            [0],
            [0],
            [],
            [1],
            [float("1")],
        )
        del set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 100]

        # pd_op.tile: (2x100xi32) <- (2x1xi32, 2xi64)
        tile_0 = paddle._C_ops.tile(set_value_with_tensor__2, full_int_array_4)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [1, 100, 1]

        # pd_op.tile: (2x100x4xf32) <- (2x1x4xf32, 3xi64)
        tile_1 = paddle._C_ops.tile(set_value_with_tensor__3, full_int_array_5)
        del full_int_array_5

        # pd_op.tile: (2x100xf32) <- (2x1xf32, 2xi64)
        tile_2 = paddle._C_ops.tile(set_value__1, full_int_array_4)

        # pd_op.nonzero: (-1x2xi64) <- (2x100xf32)
        nonzero_0 = paddle._C_ops.nonzero(tile_2)

        # pd_op.slice: (-1xi64) <- (-1x2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            nonzero_0, [1], full_int_array_2, full_int_array_3, [1], [1]
        )
        del nonzero_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [100, 100]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_55 = full_4

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_56 = full_4

        # pd_op.split: ([-1xi64, -1xi64]) <- (-1xi64, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(slice_0, full_int_array_6, full_4)
        del full_int_array_6, slice_0

        # builtin.split: (-1xi64, -1xi64) <- ([-1xi64, -1xi64])
        (
            split_0,
            split_1,
        ) = split_2
        del split_2

        # pd_op.flatten: (200xi32) <- (2x100xi32)
        flatten_3 = paddle._C_ops.flatten(tile_0, 0, 1)
        del tile_0

        # pd_op.assign: (200xi32) <- (200xi32)
        assign_57 = flatten_3
        del flatten_3

        # pd_op.flatten: (200xf32) <- (2x100xf32)
        flatten_4 = paddle._C_ops.flatten(tile_2, 0, 1)
        del tile_2

        # pd_op.assign: (200xf32) <- (200xf32)
        assign_58 = flatten_4
        del flatten_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [200]

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_59 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_60 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_61 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_62 = full_5

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_63 = full_5

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_64 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_65 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_66 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_67 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_68 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_69 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_70 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_71 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_72 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_73 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_74 = full_6

        # pd_op.uniform: (200xf32) <- (1xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_7,
            paddle.float32,
            full_5,
            full_6,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_int_array_7

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.25"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (200xb) <- (200xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(uniform_0, full_7)
        del full_7, uniform_0

        # pd_op.cast: (200xf32) <- (200xb)
        cast_0 = paddle._C_ops.cast(less_than_0, paddle.float32)
        del less_than_0

        # pd_op.multiply: (200xf32) <- (200xf32, 200xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, assign_58)
        del cast_0

        # pd_op.nonzero: (-1x1xi64) <- (200xf32)
        nonzero_1 = paddle._C_ops.nonzero(multiply_0)
        del multiply_0

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_2 = paddle._C_ops.squeeze(nonzero_1, full_int_array_0)
        del nonzero_1

        # pd_op.shape64: (1xi64) <- (-1xi64)
        shape64_0 = paddle._C_ops.shape64(squeeze_2)

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (1xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [slice_2]
        del slice_2

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_3 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.randint: (-1xi64) <- (1xi64)
        randint_0 = paddle._C_ops.randint(
            0, 2, stack_3, paddle.int64, paddle.framework._current_expected_place()
        )
        del stack_3

        # pd_op.cast: (-1xi32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(randint_0, paddle.int32)
        del randint_0

        # pd_op.scatter: (200xi32) <- (200xi32, -1xi64, -1xi32)
        scatter_0 = paddle._C_ops.scatter(assign_57, squeeze_2, cast_1, True)
        del cast_1, squeeze_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [2, 100]

        # pd_op.reshape: (2x100xi32) <- (200xi32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(assign_57, full_int_array_8)

        # pd_op.reshape: (2x100xf32) <- (200xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(assign_58, full_int_array_8)
        del assign_58

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_75 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_76 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_77 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_78 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_79 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_80 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_81 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_82 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_83 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_84 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_85 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_86 = full_int_array_9

        # pd_op.slice: (2x100x2xf32) <- (2x100x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            tile_1, [2], full_int_array_3, full_int_array_9, [1], []
        )

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_87 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_88 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_89 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_90 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_91 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_92 = full_8

        # pd_op.scale: (2x100x2xf32) <- (2x100x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_3, full_8, float("0"), True)
        del slice_3

        # pd_op.slice: (2x100x2xf32) <- (2x100x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            tile_1, [2], full_int_array_3, full_int_array_9, [1], []
        )

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([2x100x2xf32, 2x100x2xf32]) <- (2x100x2xf32, 2x100x2xf32)
        combine_2 = [scale_0, slice_4]
        del scale_0, slice_4

        # pd_op.concat: (2x100x4xf32) <- ([2x100x2xf32, 2x100x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_9)
        del combine_2

        # pd_op.scale: (2x100x4xf32) <- (2x100x4xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(concat_1, full_6, float("0"), True)
        del concat_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [2, 100, 4]

        # pd_op.uniform: (2x100x4xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_10,
            paddle.float32,
            full_5,
            full_6,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_int_array_10

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_93 = full_10

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_94 = full_10

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_95 = full_10

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_96 = full_10

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_97 = full_10

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_98 = full_10

        # pd_op.scale: (2x100x4xf32) <- (2x100x4xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(uniform_1, full_10, float("0"), True)
        del uniform_1

        # pd_op.scale: (2x100x4xf32) <- (2x100x4xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_6, float("-1"), True)
        del scale_2

        # pd_op.multiply: (2x100x4xf32) <- (2x100x4xf32, 2x100x4xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, scale_3)
        del scale_1, scale_3

        # pd_op.add: (2x100x4xf32) <- (2x100x4xf32, 2x100x4xf32)
        add_2 = paddle._C_ops.add(tile_1, multiply_1)
        del multiply_1, tile_1

        # pd_op.clip: (2x100x4xf32) <- (2x100x4xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(add_2, full_5, full_6)
        del add_2

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("1e-05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_99 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_100 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_101 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_102 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_103 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_104 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_105 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_106 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_107 = full_11

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_108 = full_11

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_109 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_110 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_111 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_112 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_113 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_114 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_115 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_116 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_117 = full_12

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_118 = full_12

        # pd_op.clip: (2x100x4xf32) <- (2x100x4xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(clip_0, full_11, full_12)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_119 = full_13

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_120 = full_13

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_121 = full_13

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_122 = full_13

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_123 = full_13

        # pd_op.scale: (2x100x4xf32) <- (2x100x4xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(clip_0, full_13, float("1"), True)
        del clip_0

        # pd_op.clip: (2x100x4xf32) <- (2x100x4xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(scale_4, full_11, full_12)
        del scale_4

        # pd_op.divide: (2x100x4xf32) <- (2x100x4xf32, 2x100x4xf32)
        divide_0 = paddle._C_ops.divide(clip_1, clip_2)
        del clip_1, clip_2

        # pd_op.log: (2x100x4xf32) <- (2x100x4xf32)
        log_0 = paddle._C_ops.log(divide_0)
        del divide_0

        # pd_op.full: (1x256xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1, 256],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # builtin.combine: ([2x256xf32, 1x256xf32]) <- (2x256xf32, 1x256xf32)
        combine_3 = [data_0, full_14]
        del data_0

        # pd_op.concat: (3x256xf32) <- ([2x256xf32, 1x256xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_4)
        del combine_3

        # pd_op.flatten: (200xi32) <- (200xi32)
        flatten_5 = paddle._C_ops.flatten(assign_57, 0, 0)
        del assign_57

        # pd_op.gather: (200x256xf32) <- (3x256xf32, 200xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(concat_2, flatten_5, full_4)
        del full_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [2, 100, -1]

        # pd_op.reshape: (2x100x256xf32) <- (200x256xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(gather_0, full_int_array_11)
        del full_int_array_11

        # pd_op.full: (400x400xf32) <- ()
        full_15 = paddle._C_ops.full(
            [400, 400],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (400x400xb) <- (400x400xf32, xf32)
        less_than_1 = paddle._C_ops.less_than(full_15, full_16)
        del full_15

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [100, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [2147483647, 100]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [1, 1]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            less_than_1,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_12, full_int_array_13, less_than_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [0, 1]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_15,
            full_int_array_4,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__2

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__4 = paddle._C_ops.set_value_(
            set_value__3,
            full_int_array_15,
            full_int_array_4,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_15, full_int_array_4, set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [1, 0]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_16,
            full_int_array_17,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_16, set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [1, 2]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_18,
            full_int_array_8,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_18, full_int_array_8, set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [2, 1]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_17,
            full_int_array_19,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_17, full_int_array_19, set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [2, 3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [3, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_20,
            full_int_array_21,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_20, full_int_array_21, set_value__7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [2, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [3, 2]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__9 = paddle._C_ops.set_value_(
            set_value__8,
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_22, full_int_array_23, set_value__8

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [3, 4]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [4, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__10 = paddle._C_ops.set_value_(
            set_value__9,
            full_int_array_24,
            full_int_array_25,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_24, full_int_array_25, set_value__9

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_27 = [4, 3]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__11 = paddle._C_ops.set_value_(
            set_value__10,
            full_int_array_26,
            full_int_array_27,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_26, full_int_array_27, set_value__10

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [4, 5]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_29 = [5, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__12 = paddle._C_ops.set_value_(
            set_value__11,
            full_int_array_28,
            full_int_array_29,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_28, full_int_array_29, set_value__11

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_30 = [4, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [5, 4]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__13 = paddle._C_ops.set_value_(
            set_value__12,
            full_int_array_30,
            full_int_array_31,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_30, full_int_array_31, set_value__12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [5, 6]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_33 = [6, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__14 = paddle._C_ops.set_value_(
            set_value__13,
            full_int_array_32,
            full_int_array_33,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_32, full_int_array_33, set_value__13

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_34 = [5, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_35 = [6, 5]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__15 = paddle._C_ops.set_value_(
            set_value__14,
            full_int_array_34,
            full_int_array_35,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_34, full_int_array_35, set_value__14

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [6, 7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [7, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__16 = paddle._C_ops.set_value_(
            set_value__15,
            full_int_array_36,
            full_int_array_37,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_36, full_int_array_37, set_value__15

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [6, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [7, 6]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__17 = paddle._C_ops.set_value_(
            set_value__16,
            full_int_array_38,
            full_int_array_39,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_38, full_int_array_39, set_value__16

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [7, 8]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_41 = [8, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__18 = paddle._C_ops.set_value_(
            set_value__17,
            full_int_array_40,
            full_int_array_41,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_40, full_int_array_41, set_value__17

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [8, 7]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__19 = paddle._C_ops.set_value_(
            set_value__18,
            full_int_array_42,
            full_int_array_43,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_42, full_int_array_43, set_value__18

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [8, 9]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_45 = [9, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__20 = paddle._C_ops.set_value_(
            set_value__19,
            full_int_array_44,
            full_int_array_45,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_44, full_int_array_45, set_value__19

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_46 = [8, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_47 = [9, 8]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__21 = paddle._C_ops.set_value_(
            set_value__20,
            full_int_array_46,
            full_int_array_47,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_46, full_int_array_47, set_value__20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_48 = [9, 10]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_49 = [10, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__22 = paddle._C_ops.set_value_(
            set_value__21,
            full_int_array_48,
            full_int_array_49,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_48, full_int_array_49, set_value__21

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [9, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_51 = [10, 9]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__23 = paddle._C_ops.set_value_(
            set_value__22,
            full_int_array_50,
            full_int_array_51,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_50, full_int_array_51, set_value__22

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_52 = [10, 11]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_53 = [11, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__24 = paddle._C_ops.set_value_(
            set_value__23,
            full_int_array_52,
            full_int_array_53,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_52, full_int_array_53, set_value__23

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_54 = [10, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_55 = [11, 10]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__25 = paddle._C_ops.set_value_(
            set_value__24,
            full_int_array_54,
            full_int_array_55,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_54, full_int_array_55, set_value__24

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_56 = [11, 12]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_57 = [12, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__26 = paddle._C_ops.set_value_(
            set_value__25,
            full_int_array_56,
            full_int_array_57,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_56, full_int_array_57, set_value__25

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [11, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_59 = [12, 11]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__27 = paddle._C_ops.set_value_(
            set_value__26,
            full_int_array_58,
            full_int_array_59,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_58, full_int_array_59, set_value__26

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_60 = [12, 13]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_61 = [13, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__28 = paddle._C_ops.set_value_(
            set_value__27,
            full_int_array_60,
            full_int_array_61,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_60, full_int_array_61, set_value__27

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_62 = [12, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_63 = [13, 12]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__29 = paddle._C_ops.set_value_(
            set_value__28,
            full_int_array_62,
            full_int_array_63,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_62, full_int_array_63, set_value__28

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_64 = [13, 14]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_65 = [14, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__30 = paddle._C_ops.set_value_(
            set_value__29,
            full_int_array_64,
            full_int_array_65,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_64, full_int_array_65, set_value__29

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_66 = [13, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_67 = [14, 13]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__31 = paddle._C_ops.set_value_(
            set_value__30,
            full_int_array_66,
            full_int_array_67,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_66, full_int_array_67, set_value__30

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_68 = [14, 15]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_69 = [15, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__32 = paddle._C_ops.set_value_(
            set_value__31,
            full_int_array_68,
            full_int_array_69,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_68, full_int_array_69, set_value__31

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_70 = [14, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_71 = [15, 14]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__33 = paddle._C_ops.set_value_(
            set_value__32,
            full_int_array_70,
            full_int_array_71,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_70, full_int_array_71, set_value__32

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_72 = [15, 16]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_73 = [16, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__34 = paddle._C_ops.set_value_(
            set_value__33,
            full_int_array_72,
            full_int_array_73,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_72, full_int_array_73, set_value__33

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_74 = [15, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_75 = [16, 15]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__35 = paddle._C_ops.set_value_(
            set_value__34,
            full_int_array_74,
            full_int_array_75,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_74, full_int_array_75, set_value__34

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_76 = [16, 17]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_77 = [17, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__36 = paddle._C_ops.set_value_(
            set_value__35,
            full_int_array_76,
            full_int_array_77,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_76, full_int_array_77, set_value__35

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_78 = [16, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_79 = [17, 16]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__37 = paddle._C_ops.set_value_(
            set_value__36,
            full_int_array_78,
            full_int_array_79,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_78, full_int_array_79, set_value__36

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_80 = [17, 18]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_81 = [18, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__38 = paddle._C_ops.set_value_(
            set_value__37,
            full_int_array_80,
            full_int_array_81,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_80, full_int_array_81, set_value__37

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_82 = [17, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_83 = [18, 17]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__39 = paddle._C_ops.set_value_(
            set_value__38,
            full_int_array_82,
            full_int_array_83,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_82, full_int_array_83, set_value__38

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_84 = [18, 19]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_85 = [19, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__40 = paddle._C_ops.set_value_(
            set_value__39,
            full_int_array_84,
            full_int_array_85,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_84, full_int_array_85, set_value__39

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_86 = [18, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_87 = [19, 18]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__41 = paddle._C_ops.set_value_(
            set_value__40,
            full_int_array_86,
            full_int_array_87,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_86, full_int_array_87, set_value__40

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_88 = [19, 20]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_89 = [20, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__42 = paddle._C_ops.set_value_(
            set_value__41,
            full_int_array_88,
            full_int_array_89,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_88, full_int_array_89, set_value__41

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_90 = [19, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_91 = [20, 19]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__43 = paddle._C_ops.set_value_(
            set_value__42,
            full_int_array_90,
            full_int_array_91,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_90, full_int_array_91, set_value__42

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_92 = [20, 21]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_93 = [21, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__44 = paddle._C_ops.set_value_(
            set_value__43,
            full_int_array_92,
            full_int_array_93,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_92, full_int_array_93, set_value__43

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_94 = [20, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_95 = [21, 20]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__45 = paddle._C_ops.set_value_(
            set_value__44,
            full_int_array_94,
            full_int_array_95,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_94, full_int_array_95, set_value__44

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_96 = [21, 22]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_97 = [22, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__46 = paddle._C_ops.set_value_(
            set_value__45,
            full_int_array_96,
            full_int_array_97,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_96, full_int_array_97, set_value__45

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_98 = [21, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_99 = [22, 21]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__47 = paddle._C_ops.set_value_(
            set_value__46,
            full_int_array_98,
            full_int_array_99,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_98, full_int_array_99, set_value__46

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_100 = [22, 23]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_101 = [23, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__48 = paddle._C_ops.set_value_(
            set_value__47,
            full_int_array_100,
            full_int_array_101,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_100, full_int_array_101, set_value__47

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_102 = [22, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_103 = [23, 22]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__49 = paddle._C_ops.set_value_(
            set_value__48,
            full_int_array_102,
            full_int_array_103,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_102, full_int_array_103, set_value__48

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_104 = [23, 24]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_105 = [24, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__50 = paddle._C_ops.set_value_(
            set_value__49,
            full_int_array_104,
            full_int_array_105,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_104, full_int_array_105, set_value__49

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_106 = [23, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_107 = [24, 23]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__51 = paddle._C_ops.set_value_(
            set_value__50,
            full_int_array_106,
            full_int_array_107,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_106, full_int_array_107, set_value__50

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_108 = [24, 25]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_109 = [25, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__52 = paddle._C_ops.set_value_(
            set_value__51,
            full_int_array_108,
            full_int_array_109,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_108, full_int_array_109, set_value__51

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_110 = [24, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_111 = [25, 24]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__53 = paddle._C_ops.set_value_(
            set_value__52,
            full_int_array_110,
            full_int_array_111,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_110, full_int_array_111, set_value__52

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_112 = [25, 26]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_113 = [26, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__54 = paddle._C_ops.set_value_(
            set_value__53,
            full_int_array_112,
            full_int_array_113,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_112, full_int_array_113, set_value__53

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_114 = [25, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_115 = [26, 25]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__55 = paddle._C_ops.set_value_(
            set_value__54,
            full_int_array_114,
            full_int_array_115,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_114, full_int_array_115, set_value__54

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_116 = [26, 27]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_117 = [27, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__56 = paddle._C_ops.set_value_(
            set_value__55,
            full_int_array_116,
            full_int_array_117,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_116, full_int_array_117, set_value__55

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_118 = [26, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_119 = [27, 26]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__57 = paddle._C_ops.set_value_(
            set_value__56,
            full_int_array_118,
            full_int_array_119,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_118, full_int_array_119, set_value__56

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_120 = [27, 28]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_121 = [28, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__58 = paddle._C_ops.set_value_(
            set_value__57,
            full_int_array_120,
            full_int_array_121,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_120, full_int_array_121, set_value__57

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_122 = [27, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_123 = [28, 27]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__59 = paddle._C_ops.set_value_(
            set_value__58,
            full_int_array_122,
            full_int_array_123,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_122, full_int_array_123, set_value__58

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_124 = [28, 29]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_125 = [29, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__60 = paddle._C_ops.set_value_(
            set_value__59,
            full_int_array_124,
            full_int_array_125,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_124, full_int_array_125, set_value__59

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_126 = [28, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_127 = [29, 28]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__61 = paddle._C_ops.set_value_(
            set_value__60,
            full_int_array_126,
            full_int_array_127,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_126, full_int_array_127, set_value__60

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_128 = [29, 30]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_129 = [30, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__62 = paddle._C_ops.set_value_(
            set_value__61,
            full_int_array_128,
            full_int_array_129,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_128, full_int_array_129, set_value__61

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_130 = [29, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_131 = [30, 29]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__63 = paddle._C_ops.set_value_(
            set_value__62,
            full_int_array_130,
            full_int_array_131,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_130, full_int_array_131, set_value__62

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_132 = [30, 31]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_133 = [31, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__64 = paddle._C_ops.set_value_(
            set_value__63,
            full_int_array_132,
            full_int_array_133,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_132, full_int_array_133, set_value__63

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_134 = [30, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_135 = [31, 30]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__65 = paddle._C_ops.set_value_(
            set_value__64,
            full_int_array_134,
            full_int_array_135,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_134, full_int_array_135, set_value__64

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_136 = [31, 32]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_137 = [32, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__66 = paddle._C_ops.set_value_(
            set_value__65,
            full_int_array_136,
            full_int_array_137,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_136, full_int_array_137, set_value__65

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_138 = [31, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_139 = [32, 31]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__67 = paddle._C_ops.set_value_(
            set_value__66,
            full_int_array_138,
            full_int_array_139,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_138, full_int_array_139, set_value__66

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_140 = [32, 33]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_141 = [33, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__68 = paddle._C_ops.set_value_(
            set_value__67,
            full_int_array_140,
            full_int_array_141,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_140, full_int_array_141, set_value__67

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_142 = [32, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_143 = [33, 32]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__69 = paddle._C_ops.set_value_(
            set_value__68,
            full_int_array_142,
            full_int_array_143,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_142, full_int_array_143, set_value__68

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_144 = [33, 34]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_145 = [34, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__70 = paddle._C_ops.set_value_(
            set_value__69,
            full_int_array_144,
            full_int_array_145,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_144, full_int_array_145, set_value__69

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_146 = [33, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_147 = [34, 33]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__71 = paddle._C_ops.set_value_(
            set_value__70,
            full_int_array_146,
            full_int_array_147,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_146, full_int_array_147, set_value__70

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_148 = [34, 35]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_149 = [35, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__72 = paddle._C_ops.set_value_(
            set_value__71,
            full_int_array_148,
            full_int_array_149,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_148, full_int_array_149, set_value__71

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_150 = [34, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_151 = [35, 34]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__73 = paddle._C_ops.set_value_(
            set_value__72,
            full_int_array_150,
            full_int_array_151,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_150, full_int_array_151, set_value__72

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_152 = [35, 36]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_153 = [36, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__74 = paddle._C_ops.set_value_(
            set_value__73,
            full_int_array_152,
            full_int_array_153,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_152, full_int_array_153, set_value__73

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_154 = [35, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_155 = [36, 35]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__75 = paddle._C_ops.set_value_(
            set_value__74,
            full_int_array_154,
            full_int_array_155,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_154, full_int_array_155, set_value__74

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_156 = [36, 37]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_157 = [37, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__76 = paddle._C_ops.set_value_(
            set_value__75,
            full_int_array_156,
            full_int_array_157,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_156, full_int_array_157, set_value__75

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_158 = [36, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_159 = [37, 36]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__77 = paddle._C_ops.set_value_(
            set_value__76,
            full_int_array_158,
            full_int_array_159,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_158, full_int_array_159, set_value__76

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_160 = [37, 38]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_161 = [38, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__78 = paddle._C_ops.set_value_(
            set_value__77,
            full_int_array_160,
            full_int_array_161,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_160, full_int_array_161, set_value__77

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_162 = [37, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_163 = [38, 37]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__79 = paddle._C_ops.set_value_(
            set_value__78,
            full_int_array_162,
            full_int_array_163,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_162, full_int_array_163, set_value__78

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_164 = [38, 39]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_165 = [39, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__80 = paddle._C_ops.set_value_(
            set_value__79,
            full_int_array_164,
            full_int_array_165,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_164, full_int_array_165, set_value__79

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_166 = [38, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_167 = [39, 38]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__81 = paddle._C_ops.set_value_(
            set_value__80,
            full_int_array_166,
            full_int_array_167,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_166, full_int_array_167, set_value__80

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_168 = [39, 40]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_169 = [40, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__82 = paddle._C_ops.set_value_(
            set_value__81,
            full_int_array_168,
            full_int_array_169,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_168, full_int_array_169, set_value__81

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_170 = [39, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_171 = [40, 39]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__83 = paddle._C_ops.set_value_(
            set_value__82,
            full_int_array_170,
            full_int_array_171,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_170, full_int_array_171, set_value__82

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_172 = [40, 41]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_173 = [41, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__84 = paddle._C_ops.set_value_(
            set_value__83,
            full_int_array_172,
            full_int_array_173,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_172, full_int_array_173, set_value__83

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_174 = [40, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_175 = [41, 40]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__85 = paddle._C_ops.set_value_(
            set_value__84,
            full_int_array_174,
            full_int_array_175,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_174, full_int_array_175, set_value__84

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_176 = [41, 42]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_177 = [42, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__86 = paddle._C_ops.set_value_(
            set_value__85,
            full_int_array_176,
            full_int_array_177,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_176, full_int_array_177, set_value__85

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_178 = [41, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_179 = [42, 41]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__87 = paddle._C_ops.set_value_(
            set_value__86,
            full_int_array_178,
            full_int_array_179,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_178, full_int_array_179, set_value__86

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_180 = [42, 43]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_181 = [43, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__88 = paddle._C_ops.set_value_(
            set_value__87,
            full_int_array_180,
            full_int_array_181,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_180, full_int_array_181, set_value__87

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_182 = [42, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_183 = [43, 42]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__89 = paddle._C_ops.set_value_(
            set_value__88,
            full_int_array_182,
            full_int_array_183,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_182, full_int_array_183, set_value__88

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_184 = [43, 44]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_185 = [44, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__90 = paddle._C_ops.set_value_(
            set_value__89,
            full_int_array_184,
            full_int_array_185,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_184, full_int_array_185, set_value__89

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_186 = [43, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_187 = [44, 43]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__91 = paddle._C_ops.set_value_(
            set_value__90,
            full_int_array_186,
            full_int_array_187,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_186, full_int_array_187, set_value__90

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_188 = [44, 45]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_189 = [45, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__92 = paddle._C_ops.set_value_(
            set_value__91,
            full_int_array_188,
            full_int_array_189,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_188, full_int_array_189, set_value__91

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_190 = [44, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_191 = [45, 44]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__93 = paddle._C_ops.set_value_(
            set_value__92,
            full_int_array_190,
            full_int_array_191,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_190, full_int_array_191, set_value__92

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_192 = [45, 46]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_193 = [46, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__94 = paddle._C_ops.set_value_(
            set_value__93,
            full_int_array_192,
            full_int_array_193,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_192, full_int_array_193, set_value__93

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_194 = [45, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_195 = [46, 45]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__95 = paddle._C_ops.set_value_(
            set_value__94,
            full_int_array_194,
            full_int_array_195,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_194, full_int_array_195, set_value__94

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_196 = [46, 47]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_197 = [47, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__96 = paddle._C_ops.set_value_(
            set_value__95,
            full_int_array_196,
            full_int_array_197,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_196, full_int_array_197, set_value__95

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_198 = [46, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_199 = [47, 46]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__97 = paddle._C_ops.set_value_(
            set_value__96,
            full_int_array_198,
            full_int_array_199,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_198, full_int_array_199, set_value__96

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_200 = [47, 48]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_201 = [48, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__98 = paddle._C_ops.set_value_(
            set_value__97,
            full_int_array_200,
            full_int_array_201,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_200, full_int_array_201, set_value__97

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_202 = [47, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_203 = [48, 47]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__99 = paddle._C_ops.set_value_(
            set_value__98,
            full_int_array_202,
            full_int_array_203,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_202, full_int_array_203, set_value__98

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_204 = [48, 49]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_205 = [49, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__100 = paddle._C_ops.set_value_(
            set_value__99,
            full_int_array_204,
            full_int_array_205,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_204, full_int_array_205, set_value__99

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_206 = [48, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_207 = [49, 48]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__101 = paddle._C_ops.set_value_(
            set_value__100,
            full_int_array_206,
            full_int_array_207,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_206, full_int_array_207, set_value__100

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_208 = [49, 50]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_209 = [50, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__102 = paddle._C_ops.set_value_(
            set_value__101,
            full_int_array_208,
            full_int_array_209,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_208, full_int_array_209, set_value__101

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_210 = [49, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_211 = [50, 49]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__103 = paddle._C_ops.set_value_(
            set_value__102,
            full_int_array_210,
            full_int_array_211,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_210, full_int_array_211, set_value__102

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_212 = [50, 51]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_213 = [51, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__104 = paddle._C_ops.set_value_(
            set_value__103,
            full_int_array_212,
            full_int_array_213,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_212, full_int_array_213, set_value__103

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_214 = [50, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_215 = [51, 50]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__105 = paddle._C_ops.set_value_(
            set_value__104,
            full_int_array_214,
            full_int_array_215,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_214, full_int_array_215, set_value__104

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_216 = [51, 52]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_217 = [52, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__106 = paddle._C_ops.set_value_(
            set_value__105,
            full_int_array_216,
            full_int_array_217,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_216, full_int_array_217, set_value__105

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_218 = [51, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_219 = [52, 51]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__107 = paddle._C_ops.set_value_(
            set_value__106,
            full_int_array_218,
            full_int_array_219,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_218, full_int_array_219, set_value__106

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_220 = [52, 53]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_221 = [53, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__108 = paddle._C_ops.set_value_(
            set_value__107,
            full_int_array_220,
            full_int_array_221,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_220, full_int_array_221, set_value__107

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_222 = [52, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_223 = [53, 52]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__109 = paddle._C_ops.set_value_(
            set_value__108,
            full_int_array_222,
            full_int_array_223,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_222, full_int_array_223, set_value__108

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_224 = [53, 54]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_225 = [54, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__110 = paddle._C_ops.set_value_(
            set_value__109,
            full_int_array_224,
            full_int_array_225,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_224, full_int_array_225, set_value__109

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_226 = [53, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_227 = [54, 53]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__111 = paddle._C_ops.set_value_(
            set_value__110,
            full_int_array_226,
            full_int_array_227,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_226, full_int_array_227, set_value__110

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_228 = [54, 55]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_229 = [55, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__112 = paddle._C_ops.set_value_(
            set_value__111,
            full_int_array_228,
            full_int_array_229,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_228, full_int_array_229, set_value__111

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_230 = [54, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_231 = [55, 54]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__113 = paddle._C_ops.set_value_(
            set_value__112,
            full_int_array_230,
            full_int_array_231,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_230, full_int_array_231, set_value__112

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_232 = [55, 56]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_233 = [56, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__114 = paddle._C_ops.set_value_(
            set_value__113,
            full_int_array_232,
            full_int_array_233,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_232, full_int_array_233, set_value__113

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_234 = [55, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_235 = [56, 55]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__115 = paddle._C_ops.set_value_(
            set_value__114,
            full_int_array_234,
            full_int_array_235,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_234, full_int_array_235, set_value__114

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_236 = [56, 57]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_237 = [57, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__116 = paddle._C_ops.set_value_(
            set_value__115,
            full_int_array_236,
            full_int_array_237,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_236, full_int_array_237, set_value__115

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_238 = [56, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_239 = [57, 56]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__117 = paddle._C_ops.set_value_(
            set_value__116,
            full_int_array_238,
            full_int_array_239,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_238, full_int_array_239, set_value__116

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_240 = [57, 58]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_241 = [58, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__118 = paddle._C_ops.set_value_(
            set_value__117,
            full_int_array_240,
            full_int_array_241,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_240, full_int_array_241, set_value__117

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_242 = [57, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_243 = [58, 57]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__119 = paddle._C_ops.set_value_(
            set_value__118,
            full_int_array_242,
            full_int_array_243,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_242, full_int_array_243, set_value__118

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_244 = [58, 59]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_245 = [59, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__120 = paddle._C_ops.set_value_(
            set_value__119,
            full_int_array_244,
            full_int_array_245,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_244, full_int_array_245, set_value__119

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_246 = [58, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_247 = [59, 58]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__121 = paddle._C_ops.set_value_(
            set_value__120,
            full_int_array_246,
            full_int_array_247,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_246, full_int_array_247, set_value__120

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_248 = [59, 60]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_249 = [60, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__122 = paddle._C_ops.set_value_(
            set_value__121,
            full_int_array_248,
            full_int_array_249,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_248, full_int_array_249, set_value__121

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_250 = [59, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_251 = [60, 59]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__123 = paddle._C_ops.set_value_(
            set_value__122,
            full_int_array_250,
            full_int_array_251,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_250, full_int_array_251, set_value__122

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_252 = [60, 61]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_253 = [61, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__124 = paddle._C_ops.set_value_(
            set_value__123,
            full_int_array_252,
            full_int_array_253,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_252, full_int_array_253, set_value__123

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_254 = [60, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_255 = [61, 60]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__125 = paddle._C_ops.set_value_(
            set_value__124,
            full_int_array_254,
            full_int_array_255,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_254, full_int_array_255, set_value__124

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_256 = [61, 62]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_257 = [62, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__126 = paddle._C_ops.set_value_(
            set_value__125,
            full_int_array_256,
            full_int_array_257,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_256, full_int_array_257, set_value__125

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_258 = [61, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_259 = [62, 61]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__127 = paddle._C_ops.set_value_(
            set_value__126,
            full_int_array_258,
            full_int_array_259,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_258, full_int_array_259, set_value__126

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_260 = [62, 63]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_261 = [63, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__128 = paddle._C_ops.set_value_(
            set_value__127,
            full_int_array_260,
            full_int_array_261,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_260, full_int_array_261, set_value__127

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_262 = [62, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_263 = [63, 62]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__129 = paddle._C_ops.set_value_(
            set_value__128,
            full_int_array_262,
            full_int_array_263,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_262, full_int_array_263, set_value__128

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_264 = [63, 64]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_265 = [64, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__130 = paddle._C_ops.set_value_(
            set_value__129,
            full_int_array_264,
            full_int_array_265,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_264, full_int_array_265, set_value__129

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_266 = [63, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_267 = [64, 63]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__131 = paddle._C_ops.set_value_(
            set_value__130,
            full_int_array_266,
            full_int_array_267,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_266, full_int_array_267, set_value__130

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_268 = [64, 65]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_269 = [65, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__132 = paddle._C_ops.set_value_(
            set_value__131,
            full_int_array_268,
            full_int_array_269,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_268, full_int_array_269, set_value__131

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_270 = [64, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_271 = [65, 64]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__133 = paddle._C_ops.set_value_(
            set_value__132,
            full_int_array_270,
            full_int_array_271,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_270, full_int_array_271, set_value__132

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_272 = [65, 66]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_273 = [66, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__134 = paddle._C_ops.set_value_(
            set_value__133,
            full_int_array_272,
            full_int_array_273,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_272, full_int_array_273, set_value__133

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_274 = [65, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_275 = [66, 65]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__135 = paddle._C_ops.set_value_(
            set_value__134,
            full_int_array_274,
            full_int_array_275,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_274, full_int_array_275, set_value__134

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_276 = [66, 67]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_277 = [67, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__136 = paddle._C_ops.set_value_(
            set_value__135,
            full_int_array_276,
            full_int_array_277,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_276, full_int_array_277, set_value__135

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_278 = [66, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_279 = [67, 66]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__137 = paddle._C_ops.set_value_(
            set_value__136,
            full_int_array_278,
            full_int_array_279,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_278, full_int_array_279, set_value__136

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_280 = [67, 68]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_281 = [68, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__138 = paddle._C_ops.set_value_(
            set_value__137,
            full_int_array_280,
            full_int_array_281,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_280, full_int_array_281, set_value__137

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_282 = [67, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_283 = [68, 67]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__139 = paddle._C_ops.set_value_(
            set_value__138,
            full_int_array_282,
            full_int_array_283,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_282, full_int_array_283, set_value__138

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_284 = [68, 69]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_285 = [69, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__140 = paddle._C_ops.set_value_(
            set_value__139,
            full_int_array_284,
            full_int_array_285,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_284, full_int_array_285, set_value__139

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_286 = [68, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_287 = [69, 68]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__141 = paddle._C_ops.set_value_(
            set_value__140,
            full_int_array_286,
            full_int_array_287,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_286, full_int_array_287, set_value__140

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_288 = [69, 70]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_289 = [70, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__142 = paddle._C_ops.set_value_(
            set_value__141,
            full_int_array_288,
            full_int_array_289,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_288, full_int_array_289, set_value__141

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_290 = [69, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_291 = [70, 69]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__143 = paddle._C_ops.set_value_(
            set_value__142,
            full_int_array_290,
            full_int_array_291,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_290, full_int_array_291, set_value__142

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_292 = [70, 71]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_293 = [71, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__144 = paddle._C_ops.set_value_(
            set_value__143,
            full_int_array_292,
            full_int_array_293,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_292, full_int_array_293, set_value__143

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_294 = [70, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_295 = [71, 70]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__145 = paddle._C_ops.set_value_(
            set_value__144,
            full_int_array_294,
            full_int_array_295,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_294, full_int_array_295, set_value__144

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_296 = [71, 72]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_297 = [72, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__146 = paddle._C_ops.set_value_(
            set_value__145,
            full_int_array_296,
            full_int_array_297,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_296, full_int_array_297, set_value__145

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_298 = [71, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_299 = [72, 71]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__147 = paddle._C_ops.set_value_(
            set_value__146,
            full_int_array_298,
            full_int_array_299,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_298, full_int_array_299, set_value__146

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_300 = [72, 73]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_301 = [73, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__148 = paddle._C_ops.set_value_(
            set_value__147,
            full_int_array_300,
            full_int_array_301,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_300, full_int_array_301, set_value__147

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_302 = [72, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_303 = [73, 72]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__149 = paddle._C_ops.set_value_(
            set_value__148,
            full_int_array_302,
            full_int_array_303,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_302, full_int_array_303, set_value__148

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_304 = [73, 74]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_305 = [74, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__150 = paddle._C_ops.set_value_(
            set_value__149,
            full_int_array_304,
            full_int_array_305,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_304, full_int_array_305, set_value__149

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_306 = [73, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_307 = [74, 73]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__151 = paddle._C_ops.set_value_(
            set_value__150,
            full_int_array_306,
            full_int_array_307,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_306, full_int_array_307, set_value__150

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_308 = [74, 75]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_309 = [75, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__152 = paddle._C_ops.set_value_(
            set_value__151,
            full_int_array_308,
            full_int_array_309,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_308, full_int_array_309, set_value__151

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_310 = [74, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_311 = [75, 74]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__153 = paddle._C_ops.set_value_(
            set_value__152,
            full_int_array_310,
            full_int_array_311,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_310, full_int_array_311, set_value__152

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_312 = [75, 76]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_313 = [76, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__154 = paddle._C_ops.set_value_(
            set_value__153,
            full_int_array_312,
            full_int_array_313,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_312, full_int_array_313, set_value__153

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_314 = [75, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_315 = [76, 75]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__155 = paddle._C_ops.set_value_(
            set_value__154,
            full_int_array_314,
            full_int_array_315,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_314, full_int_array_315, set_value__154

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_316 = [76, 77]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_317 = [77, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__156 = paddle._C_ops.set_value_(
            set_value__155,
            full_int_array_316,
            full_int_array_317,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_316, full_int_array_317, set_value__155

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_318 = [76, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_319 = [77, 76]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__157 = paddle._C_ops.set_value_(
            set_value__156,
            full_int_array_318,
            full_int_array_319,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_318, full_int_array_319, set_value__156

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_320 = [77, 78]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_321 = [78, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__158 = paddle._C_ops.set_value_(
            set_value__157,
            full_int_array_320,
            full_int_array_321,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_320, full_int_array_321, set_value__157

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_322 = [77, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_323 = [78, 77]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__159 = paddle._C_ops.set_value_(
            set_value__158,
            full_int_array_322,
            full_int_array_323,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_322, full_int_array_323, set_value__158

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_324 = [78, 79]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_325 = [79, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__160 = paddle._C_ops.set_value_(
            set_value__159,
            full_int_array_324,
            full_int_array_325,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_324, full_int_array_325, set_value__159

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_326 = [78, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_327 = [79, 78]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__161 = paddle._C_ops.set_value_(
            set_value__160,
            full_int_array_326,
            full_int_array_327,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_326, full_int_array_327, set_value__160

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_328 = [79, 80]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_329 = [80, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__162 = paddle._C_ops.set_value_(
            set_value__161,
            full_int_array_328,
            full_int_array_329,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_328, full_int_array_329, set_value__161

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_330 = [79, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_331 = [80, 79]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__163 = paddle._C_ops.set_value_(
            set_value__162,
            full_int_array_330,
            full_int_array_331,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_330, full_int_array_331, set_value__162

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_332 = [80, 81]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_333 = [81, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__164 = paddle._C_ops.set_value_(
            set_value__163,
            full_int_array_332,
            full_int_array_333,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_332, full_int_array_333, set_value__163

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_334 = [80, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_335 = [81, 80]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__165 = paddle._C_ops.set_value_(
            set_value__164,
            full_int_array_334,
            full_int_array_335,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_334, full_int_array_335, set_value__164

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_336 = [81, 82]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_337 = [82, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__166 = paddle._C_ops.set_value_(
            set_value__165,
            full_int_array_336,
            full_int_array_337,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_336, full_int_array_337, set_value__165

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_338 = [81, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_339 = [82, 81]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__167 = paddle._C_ops.set_value_(
            set_value__166,
            full_int_array_338,
            full_int_array_339,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_338, full_int_array_339, set_value__166

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_340 = [82, 83]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_341 = [83, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__168 = paddle._C_ops.set_value_(
            set_value__167,
            full_int_array_340,
            full_int_array_341,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_340, full_int_array_341, set_value__167

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_342 = [82, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_343 = [83, 82]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__169 = paddle._C_ops.set_value_(
            set_value__168,
            full_int_array_342,
            full_int_array_343,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_342, full_int_array_343, set_value__168

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_344 = [83, 84]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_345 = [84, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__170 = paddle._C_ops.set_value_(
            set_value__169,
            full_int_array_344,
            full_int_array_345,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_344, full_int_array_345, set_value__169

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_346 = [83, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_347 = [84, 83]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__171 = paddle._C_ops.set_value_(
            set_value__170,
            full_int_array_346,
            full_int_array_347,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_346, full_int_array_347, set_value__170

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_348 = [84, 85]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_349 = [85, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__172 = paddle._C_ops.set_value_(
            set_value__171,
            full_int_array_348,
            full_int_array_349,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_348, full_int_array_349, set_value__171

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_350 = [84, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_351 = [85, 84]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__173 = paddle._C_ops.set_value_(
            set_value__172,
            full_int_array_350,
            full_int_array_351,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_350, full_int_array_351, set_value__172

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_352 = [85, 86]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_353 = [86, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__174 = paddle._C_ops.set_value_(
            set_value__173,
            full_int_array_352,
            full_int_array_353,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_352, full_int_array_353, set_value__173

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_354 = [85, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_355 = [86, 85]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__175 = paddle._C_ops.set_value_(
            set_value__174,
            full_int_array_354,
            full_int_array_355,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_354, full_int_array_355, set_value__174

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_356 = [86, 87]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_357 = [87, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__176 = paddle._C_ops.set_value_(
            set_value__175,
            full_int_array_356,
            full_int_array_357,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_356, full_int_array_357, set_value__175

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_358 = [86, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_359 = [87, 86]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__177 = paddle._C_ops.set_value_(
            set_value__176,
            full_int_array_358,
            full_int_array_359,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_358, full_int_array_359, set_value__176

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_360 = [87, 88]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_361 = [88, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__178 = paddle._C_ops.set_value_(
            set_value__177,
            full_int_array_360,
            full_int_array_361,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_360, full_int_array_361, set_value__177

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_362 = [87, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_363 = [88, 87]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__179 = paddle._C_ops.set_value_(
            set_value__178,
            full_int_array_362,
            full_int_array_363,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_362, full_int_array_363, set_value__178

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_364 = [88, 89]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_365 = [89, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__180 = paddle._C_ops.set_value_(
            set_value__179,
            full_int_array_364,
            full_int_array_365,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_364, full_int_array_365, set_value__179

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_366 = [88, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_367 = [89, 88]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__181 = paddle._C_ops.set_value_(
            set_value__180,
            full_int_array_366,
            full_int_array_367,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_366, full_int_array_367, set_value__180

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_368 = [89, 90]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_369 = [90, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__182 = paddle._C_ops.set_value_(
            set_value__181,
            full_int_array_368,
            full_int_array_369,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_368, full_int_array_369, set_value__181

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_370 = [89, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_371 = [90, 89]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__183 = paddle._C_ops.set_value_(
            set_value__182,
            full_int_array_370,
            full_int_array_371,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_370, full_int_array_371, set_value__182

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_372 = [90, 91]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_373 = [91, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__184 = paddle._C_ops.set_value_(
            set_value__183,
            full_int_array_372,
            full_int_array_373,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_372, full_int_array_373, set_value__183

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_374 = [90, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_375 = [91, 90]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__185 = paddle._C_ops.set_value_(
            set_value__184,
            full_int_array_374,
            full_int_array_375,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_374, full_int_array_375, set_value__184

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_376 = [91, 92]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_377 = [92, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__186 = paddle._C_ops.set_value_(
            set_value__185,
            full_int_array_376,
            full_int_array_377,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_376, full_int_array_377, set_value__185

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_378 = [91, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_379 = [92, 91]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__187 = paddle._C_ops.set_value_(
            set_value__186,
            full_int_array_378,
            full_int_array_379,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_378, full_int_array_379, set_value__186

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_380 = [92, 93]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_381 = [93, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__188 = paddle._C_ops.set_value_(
            set_value__187,
            full_int_array_380,
            full_int_array_381,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_380, full_int_array_381, set_value__187

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_382 = [92, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_383 = [93, 92]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__189 = paddle._C_ops.set_value_(
            set_value__188,
            full_int_array_382,
            full_int_array_383,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_382, full_int_array_383, set_value__188

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_384 = [93, 94]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_385 = [94, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__190 = paddle._C_ops.set_value_(
            set_value__189,
            full_int_array_384,
            full_int_array_385,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_384, full_int_array_385, set_value__189

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_386 = [93, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_387 = [94, 93]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__191 = paddle._C_ops.set_value_(
            set_value__190,
            full_int_array_386,
            full_int_array_387,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_386, full_int_array_387, set_value__190

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_388 = [94, 95]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_389 = [95, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__192 = paddle._C_ops.set_value_(
            set_value__191,
            full_int_array_388,
            full_int_array_389,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_388, full_int_array_389, set_value__191

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_390 = [94, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_391 = [95, 94]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__193 = paddle._C_ops.set_value_(
            set_value__192,
            full_int_array_390,
            full_int_array_391,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_390, full_int_array_391, set_value__192

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_392 = [95, 96]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_393 = [96, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__194 = paddle._C_ops.set_value_(
            set_value__193,
            full_int_array_392,
            full_int_array_393,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_392, full_int_array_393, set_value__193

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_394 = [95, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_395 = [96, 95]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__195 = paddle._C_ops.set_value_(
            set_value__194,
            full_int_array_394,
            full_int_array_395,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_394, full_int_array_395, set_value__194

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_396 = [96, 97]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_397 = [97, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__196 = paddle._C_ops.set_value_(
            set_value__195,
            full_int_array_396,
            full_int_array_397,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_396, full_int_array_397, set_value__195

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_398 = [96, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_399 = [97, 96]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__197 = paddle._C_ops.set_value_(
            set_value__196,
            full_int_array_398,
            full_int_array_399,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_398, full_int_array_399, set_value__196

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_400 = [97, 98]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_401 = [98, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__198 = paddle._C_ops.set_value_(
            set_value__197,
            full_int_array_400,
            full_int_array_401,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_400, full_int_array_401, set_value__197

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_402 = [97, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_403 = [98, 97]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__199 = paddle._C_ops.set_value_(
            set_value__198,
            full_int_array_402,
            full_int_array_403,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_402, full_int_array_403, set_value__198

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_404 = [98, 99]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_405 = [99, 100]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__200 = paddle._C_ops.set_value_(
            set_value__199,
            full_int_array_404,
            full_int_array_405,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_404, full_int_array_405, set_value__199

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_406 = [98, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_407 = [99, 98]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__201 = paddle._C_ops.set_value_(
            set_value__200,
            full_int_array_406,
            full_int_array_407,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_406, full_int_array_407, set_value__200

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_408 = [99, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_409 = [100, 99]

        # pd_op.set_value_: (400x400xb) <- (400x400xb, 2xi64, 2xi64, 2xi64)
        set_value__202 = paddle._C_ops.set_value_(
            set_value__201,
            full_int_array_408,
            full_int_array_409,
            full_int_array_14,
            [0, 1],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_14, full_int_array_408, full_int_array_409, set_value__201

        # pd_op.bitwise_not: (400x400xb) <- (400x400xb)
        bitwise_not_0 = paddle._C_ops.bitwise_not(set_value__202)

        # pd_op.full: (1xf64) <- ()
        full_17 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("96"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (96xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_17, full_18, full_19, dtype="float32")
        del full_18

        # builtin.combine: ([96xf32, 96xf32]) <- (96xf32, 96xf32)
        combine_4 = [arange_0, arange_0]
        del arange_0

        # pd_op.meshgrid: ([96x96xf32, 96x96xf32]) <- ([96xf32, 96xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (96x96xf32, 96x96xf32) <- ([96x96xf32, 96x96xf32])
        (
            split_3,
            split_4,
        ) = meshgrid_0
        del meshgrid_0

        # builtin.combine: ([96x96xf32, 96x96xf32]) <- (96x96xf32, 96x96xf32)
        combine_5 = [split_4, split_3]
        del split_3, split_4

        # pd_op.stack: (96x96x2xf32) <- ([96x96xf32, 96x96xf32])
        stack_4 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.full: (2xi64) <- ()
        full_20 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_20,
            [2],
            paddle.int64,
            [float("96"), float("96")],
            paddle.framework._current_expected_place(),
        )
        del full_20

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_2 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.unsqueeze: (1x96x96x2xf32) <- (96x96x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(stack_4, full_int_array_1)
        del stack_4

        # pd_op.scale: (1x96x96x2xf32) <- (1x96x96x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(unsqueeze_0, full_6, float("0.5"), True)
        del unsqueeze_0

        # pd_op.divide: (1x96x96x2xf32) <- (1x96x96x2xf32, 2xf32)
        divide_1 = paddle._C_ops.divide(scale_5, cast_2)
        del cast_2, scale_5

        # pd_op.full_like: (1x96x96x2xf32) <- (1x96x96x2xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            divide_1, full_6, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x96x96x2xf32) <- (1x96x96x2xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(full_like_0, full_21, float("0"), True)
        del full_like_0

        # pd_op.scale: (1x96x96x2xf32) <- (1x96x96x2xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_6, float("0"), True)
        del scale_6

        # builtin.combine: ([1x96x96x2xf32, 1x96x96x2xf32]) <- (1x96x96x2xf32, 1x96x96x2xf32)
        combine_6 = [divide_1, scale_7]
        del divide_1, scale_7

        # pd_op.concat: (1x96x96x4xf32) <- ([1x96x96x2xf32, 1x96x96x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_6, full_9)
        del combine_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_410 = [-1, 9216, 4]

        # pd_op.reshape: (1x9216x4xf32) <- (1x96x96x4xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(concat_3, full_int_array_410)
        del concat_3, full_int_array_410

        # pd_op.full: (1xf64) <- ()
        full_22 = paddle._C_ops.full(
            [1], float("48"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (48xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_17, full_22, full_19, dtype="float32")
        del full_22

        # builtin.combine: ([48xf32, 48xf32]) <- (48xf32, 48xf32)
        combine_7 = [arange_1, arange_1]
        del arange_1

        # pd_op.meshgrid: ([48x48xf32, 48x48xf32]) <- ([48xf32, 48xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_7)
        del combine_7

        # builtin.split: (48x48xf32, 48x48xf32) <- ([48x48xf32, 48x48xf32])
        (
            split_5,
            split_6,
        ) = meshgrid_1
        del meshgrid_1

        # builtin.combine: ([48x48xf32, 48x48xf32]) <- (48x48xf32, 48x48xf32)
        combine_8 = [split_6, split_5]
        del split_5, split_6

        # pd_op.stack: (48x48x2xf32) <- ([48x48xf32, 48x48xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)
        del combine_8

        # pd_op.full: (2xi64) <- ()
        full_23 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_23,
            [2],
            paddle.int64,
            [float("48"), float("48")],
            paddle.framework._current_expected_place(),
        )
        del full_23

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_3 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.unsqueeze: (1x48x48x2xf32) <- (48x48x2xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(stack_5, full_int_array_1)
        del stack_5

        # pd_op.scale: (1x48x48x2xf32) <- (1x48x48x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(unsqueeze_1, full_6, float("0.5"), True)
        del unsqueeze_1

        # pd_op.divide: (1x48x48x2xf32) <- (1x48x48x2xf32, 2xf32)
        divide_2 = paddle._C_ops.divide(scale_8, cast_3)
        del cast_3, scale_8

        # pd_op.full_like: (1x48x48x2xf32) <- (1x48x48x2xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            divide_2, full_6, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x48x48x2xf32) <- (1x48x48x2xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(full_like_1, full_21, float("0"), True)
        del full_like_1

        # pd_op.scale: (1x48x48x2xf32) <- (1x48x48x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_10, float("0"), True)
        del scale_9

        # builtin.combine: ([1x48x48x2xf32, 1x48x48x2xf32]) <- (1x48x48x2xf32, 1x48x48x2xf32)
        combine_9 = [divide_2, scale_10]
        del divide_2, scale_10

        # pd_op.concat: (1x48x48x4xf32) <- ([1x48x48x2xf32, 1x48x48x2xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_9, full_9)
        del combine_9

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_411 = [-1, 2304, 4]

        # pd_op.reshape: (1x2304x4xf32) <- (1x48x48x4xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(concat_4, full_int_array_411)
        del concat_4, full_int_array_411

        # pd_op.full: (1xf64) <- ()
        full_24 = paddle._C_ops.full(
            [1], float("24"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (24xf32) <- (1xf64, 1xf64, 1xf64)
        arange_2 = paddle.arange(full_17, full_24, full_19, dtype="float32")
        del full_24

        # builtin.combine: ([24xf32, 24xf32]) <- (24xf32, 24xf32)
        combine_10 = [arange_2, arange_2]
        del arange_2

        # pd_op.meshgrid: ([24x24xf32, 24x24xf32]) <- ([24xf32, 24xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_10)
        del combine_10

        # builtin.split: (24x24xf32, 24x24xf32) <- ([24x24xf32, 24x24xf32])
        (
            split_7,
            split_8,
        ) = meshgrid_2
        del meshgrid_2

        # builtin.combine: ([24x24xf32, 24x24xf32]) <- (24x24xf32, 24x24xf32)
        combine_11 = [split_8, split_7]
        del split_7, split_8

        # pd_op.stack: (24x24x2xf32) <- ([24x24xf32, 24x24xf32])
        stack_6 = paddle._C_ops.stack(combine_11, -1)
        del combine_11

        # pd_op.full: (2xi64) <- ()
        full_25 = paddle._C_ops.full(
            [2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (2xi64) <- (2xi64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_25,
            [2],
            paddle.int64,
            [float("24"), float("24")],
            paddle.framework._current_expected_place(),
        )
        del full_25

        # pd_op.cast: (2xf32) <- (2xi64)
        cast_4 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.unsqueeze: (1x24x24x2xf32) <- (24x24x2xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(stack_6, full_int_array_1)
        del stack_6

        # pd_op.scale: (1x24x24x2xf32) <- (1x24x24x2xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(unsqueeze_2, full_6, float("0.5"), True)
        del unsqueeze_2

        # pd_op.divide: (1x24x24x2xf32) <- (1x24x24x2xf32, 2xf32)
        divide_3 = paddle._C_ops.divide(scale_11, cast_4)
        del cast_4, scale_11

        # pd_op.full_like: (1x24x24x2xf32) <- (1x24x24x2xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            divide_3, full_6, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x24x24x2xf32) <- (1x24x24x2xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(full_like_2, full_21, float("0"), True)
        del full_21, full_like_2

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x24x24x2xf32) <- (1x24x24x2xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_26, float("0"), True)
        del full_26, scale_12

        # builtin.combine: ([1x24x24x2xf32, 1x24x24x2xf32]) <- (1x24x24x2xf32, 1x24x24x2xf32)
        combine_12 = [divide_3, scale_13]
        del divide_3, scale_13

        # pd_op.concat: (1x24x24x4xf32) <- ([1x24x24x2xf32, 1x24x24x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_12, full_9)
        del combine_12

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_412 = [-1, 576, 4]

        # pd_op.reshape: (1x576x4xf32) <- (1x24x24x4xf32, 3xi64)
        reshape_6 = paddle._C_ops.reshape(concat_5, full_int_array_412)
        del concat_5, full_int_array_412

        # builtin.combine: ([1x9216x4xf32, 1x2304x4xf32, 1x576x4xf32]) <- (1x9216x4xf32, 1x2304x4xf32, 1x576x4xf32)
        combine_13 = [reshape_4, reshape_5, reshape_6]
        del reshape_4, reshape_5, reshape_6

        # pd_op.concat: (1x12096x4xf32) <- ([1x9216x4xf32, 1x2304x4xf32, 1x576x4xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_13, full_0)
        del combine_13

        # pd_op.full: (xf32) <- ()
        full_27 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x12096x4xb) <- (1x12096x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(concat_6, full_27)
        del full_27

        # pd_op.full: (xf32) <- ()
        full_28 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (1x12096x4xb) <- (1x12096x4xf32, xf32)
        less_than_2 = paddle._C_ops.less_than(concat_6, full_28)
        del full_28

        # pd_op.multiply: (1x12096x4xb) <- (1x12096x4xb, 1x12096x4xb)
        multiply_2 = paddle._C_ops.multiply(greater_than_0, less_than_2)
        del greater_than_0, less_than_2

        # pd_op.all: (1x12096x1xb) <- (1x12096x4xb)
        all_0 = paddle._C_ops.all(multiply_2, [-1], True)
        del multiply_2

        # pd_op.scale: (1x12096x4xf32) <- (1x12096x4xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(concat_6, full_13, float("1"), True)

        # pd_op.divide: (1x12096x4xf32) <- (1x12096x4xf32, 1x12096x4xf32)
        divide_4 = paddle._C_ops.divide(concat_6, scale_14)
        del concat_6, scale_14

        # pd_op.log: (1x12096x4xf32) <- (1x12096x4xf32)
        log_1 = paddle._C_ops.log(divide_4)
        del divide_4

        # pd_op.full: (xf32) <- ()
        full_29 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_29,
            [],
            paddle.float32,
            [float("inf")],
            paddle.framework._current_expected_place(),
        )
        del full_29

        # pd_op.full_like: (1x12096x4xf32) <- (1x12096x4xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            log_1, full_5, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            assign_value__3,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x12096x1xb) <- (1x12096x1xb, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            all_0, full_5, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x12096x1xf32) <- (1x12096x1xb)
        cast_5 = paddle._C_ops.cast(full_like_5, paddle.float32)
        del full_like_5

        # pd_op.cast: (1x12096x1xf32) <- (1x12096x1xb)
        cast_6 = paddle._C_ops.cast(all_0, paddle.float32)

        # pd_op.add: (1x12096x4xf32) <- (1x12096x4xf32, xf32)
        add_3 = paddle._C_ops.add(full_like_3, full_like_4)
        del full_like_3, full_like_4

        # pd_op.add: (1x12096x4xf32) <- (1x12096x4xf32, 1x12096x1xf32)
        add_4 = paddle._C_ops.add(add_3, cast_5)
        del add_3, cast_5

        # pd_op.add: (1x12096x4xf32) <- (1x12096x4xf32, 1x12096x4xf32)
        add_5 = paddle._C_ops.add(log_1, add_4)
        del log_1

        # pd_op.add: (1x12096x4xf32) <- (xf32, 1x12096x4xf32)
        add_6 = paddle._C_ops.add(assign_value__3, add_4)
        del assign_value__3

        # pd_op.add: (1x12096x4xf32) <- (1x12096x1xf32, 1x12096x4xf32)
        add_7 = paddle._C_ops.add(cast_6, add_4)
        del add_4, cast_6

        # pd_op.cast: (1x12096x4xb) <- (1x12096x4xf32)
        cast_7 = paddle._C_ops.cast(add_7, paddle.bool)
        del add_7

        # pd_op.where: (1x12096x4xf32) <- (1x12096x4xb, 1x12096x4xf32, 1x12096x4xf32)
        where_0 = paddle._C_ops.where(cast_7, add_5, add_6)
        del add_5, add_6, cast_7

        # pd_op.full: (xf32) <- ()
        full_30 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_30,
            [],
            paddle.float32,
            [float("0")],
            paddle.framework._current_expected_place(),
        )
        del full_30

        # pd_op.full_like: (2x12096x256xf32) <- (2x12096x256xf32, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            concat_0, full_5, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            assign_value__4,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x12096x1xb) <- (1x12096x1xb, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            all_0, full_5, paddle.bool, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x12096x1xf32) <- (1x12096x1xb)
        cast_8 = paddle._C_ops.cast(full_like_8, paddle.float32)
        del full_like_8

        # pd_op.cast: (1x12096x1xf32) <- (1x12096x1xb)
        cast_9 = paddle._C_ops.cast(all_0, paddle.float32)
        del all_0

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, xf32)
        add_8 = paddle._C_ops.add(full_like_6, full_like_7)
        del full_like_6, full_like_7

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 1x12096x1xf32)
        add_9 = paddle._C_ops.add(add_8, cast_8)
        del add_8, cast_8

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 2x12096x256xf32)
        add_10 = paddle._C_ops.add(concat_0, add_9)

        # pd_op.add: (2x12096x256xf32) <- (xf32, 2x12096x256xf32)
        add_11 = paddle._C_ops.add(assign_value__4, add_9)
        del assign_value__4

        # pd_op.add: (2x12096x256xf32) <- (1x12096x1xf32, 2x12096x256xf32)
        add_12 = paddle._C_ops.add(cast_9, add_9)
        del cast_9

        # pd_op.cast: (2x12096x256xb) <- (2x12096x256xf32)
        cast_10 = paddle._C_ops.cast(add_12, paddle.bool)
        del add_12

        # pd_op.where: (2x12096x256xf32) <- (2x12096x256xb, 2x12096x256xf32, 2x12096x256xf32)
        where_1 = paddle._C_ops.where(cast_10, add_10, add_11)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(where_1, parameter_143, False, False)
        del parameter_143

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_13 = paddle._C_ops.add(matmul_0, parameter_142)
        del parameter_142

        # pd_op.layer_norm: (2x12096x256xf32, 2x12096xf32, 2x12096xf32) <- (2x12096x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_140, parameter_141

        # pd_op.matmul: (2x12096x2xf32) <- (2x12096x256xf32, 256x2xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_0, parameter_139, False, False)

        # pd_op.add: (2x12096x2xf32) <- (2x12096x2xf32, 2xf32)
        add_14 = paddle._C_ops.add(matmul_1, parameter_138)
        del matmul_1

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_0, parameter_137, False, False)

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_2, parameter_136)

        # pd_op.relu: (2x12096x256xf32) <- (2x12096x256xf32)
        relu_0 = paddle._C_ops.relu(add_15)
        del add_15

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(relu_0, parameter_135, False, False)

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_16 = paddle._C_ops.add(matmul_3, parameter_134)

        # pd_op.relu: (2x12096x256xf32) <- (2x12096x256xf32)
        relu_1 = paddle._C_ops.relu(add_16)
        del add_16

        # pd_op.matmul: (2x12096x4xf32) <- (2x12096x256xf32, 256x4xf32)
        matmul_4 = paddle._C_ops.matmul(relu_1, parameter_133, False, False)

        # pd_op.add: (2x12096x4xf32) <- (2x12096x4xf32, 4xf32)
        add_17 = paddle._C_ops.add(matmul_4, parameter_132)

        # pd_op.add: (2x12096x4xf32) <- (2x12096x4xf32, 1x12096x4xf32)
        add_18 = paddle._C_ops.add(add_17, where_0)

        # pd_op.max: (2x12096xf32) <- (2x12096x2xf32, 1xi64)
        max_0 = paddle._C_ops.max(add_14, full_int_array_0, False)
        del add_14

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full(
            [1], float("300"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.topk: (2x300xf32, 2x300xi64) <- (2x12096xf32, 1xi32)
        topk_0, topk_1 = (lambda x, f: f(x))(
            paddle._C_ops.topk(max_0, full_31, 1, True, True),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_31, max_0

        # pd_op.full: (1xf64) <- ()
        full_32 = paddle._C_ops.full(
            [1], float("2"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (2xi64) <- (1xf64, 1xf64, 1xf64)
        arange_3 = paddle.arange(full_17, full_32, full_19, dtype="int64")
        del full_32

        # pd_op.unsqueeze: (2x1xi64) <- (2xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(arange_3, full_int_array_0)
        del arange_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_413 = [1, 300]

        # pd_op.tile: (2x300xi64) <- (2x1xi64, 2xi64)
        tile_3 = paddle._C_ops.tile(unsqueeze_3, full_int_array_413)
        del full_int_array_413, unsqueeze_3

        # builtin.combine: ([2x300xi64, 2x300xi64]) <- (2x300xi64, 2x300xi64)
        combine_14 = [tile_3, topk_1]
        del tile_3, topk_1

        # pd_op.stack: (2x300x2xi64) <- ([2x300xi64, 2x300xi64])
        stack_7 = paddle._C_ops.stack(combine_14, -1)
        del combine_14

        # pd_op.gather_nd: (2x300x256xf32) <- (2x12096x256xf32, 2x300x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(layer_norm_0, stack_7)

        # pd_op.gather_nd: (2x300x4xf32) <- (2x12096x4xf32, 2x300x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(add_18, stack_7)

        # pd_op.layer_norm: (2x300x256xf32, 2x300xf32, 2x300xf32) <- (2x300x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                gather_nd_0, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x300x2xf32) <- (2x300x256xf32, 256x2xf32)
        matmul_5 = paddle._C_ops.matmul(layer_norm_3, parameter_139, False, False)

        # pd_op.add: (2x300x2xf32) <- (2x300x2xf32, 2xf32)
        add_0 = paddle._C_ops.add(matmul_5, parameter_138)

        # pd_op.matmul: (2x300x256xf32) <- (2x300x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_3, parameter_129, False, False)

        # pd_op.add: (2x300x256xf32) <- (2x300x256xf32, 256xf32)
        add_19 = paddle._C_ops.add(matmul_6, parameter_128)

        # pd_op.relu: (2x300x256xf32) <- (2x300x256xf32)
        relu_2 = paddle._C_ops.relu(add_19)
        del add_19

        # pd_op.matmul: (2x300x256xf32) <- (2x300x256xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(relu_2, parameter_127, False, False)

        # pd_op.add: (2x300x256xf32) <- (2x300x256xf32, 256xf32)
        add_20 = paddle._C_ops.add(matmul_7, parameter_126)

        # pd_op.relu: (2x300x256xf32) <- (2x300x256xf32)
        relu_3 = paddle._C_ops.relu(add_20)
        del add_20

        # pd_op.matmul: (2x300x32xf32) <- (2x300x256xf32, 256x32xf32)
        matmul_8 = paddle._C_ops.matmul(relu_3, parameter_125, False, False)

        # pd_op.add: (2x300x32xf32) <- (2x300x32xf32, 32xf32)
        add_21 = paddle._C_ops.add(matmul_8, parameter_124)

        # pd_op.flatten: (2x32x36864xf32) <- (2x32x192x192xf32)
        flatten_6 = paddle._C_ops.flatten(data_16, 2, 3)
        del data_16

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_124 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_125 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_126 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_127 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_128 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_129 = flatten_6

        # pd_op.assign: (2x32x36864xf32) <- (2x32x36864xf32)
        assign_130 = flatten_6

        # pd_op.bmm: (2x300x36864xf32) <- (2x300x32xf32, 2x32x36864xf32)
        bmm_0 = paddle._C_ops.bmm(add_21, flatten_6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_414 = [2, 300, 192, 192]

        # pd_op.reshape: (2x300x192x192xf32) <- (2x300x36864xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(bmm_0, full_int_array_414)
        del full_int_array_414

        # pd_op.sigmoid: (2x300x4xf32) <- (2x300x4xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(gather_nd_1)
        del gather_nd_1

        # pd_op.share_data_: (2x300x256xf32) <- (2x300x256xf32)
        share_data__0 = gather_nd_0.detach()

        # builtin.combine: ([2x100x256xf32, 2x300x256xf32]) <- (2x100x256xf32, 2x300x256xf32)
        combine_15 = [reshape_3, share_data__0]

        # pd_op.concat: (2x400x256xf32) <- ([2x100x256xf32, 2x300x256xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_15, full_0)
        del combine_15

        # pd_op.greater_than: (2x300x192x192xb) <- (2x300x192x192xf32, xf32)
        greater_than_1 = paddle._C_ops.greater_than(reshape_7, full_16)
        del full_16

        # pd_op.full: (1xf64) <- ()
        full_33 = paddle._C_ops.full(
            [1], float("192"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (192xf32) <- (1xf64, 1xf64, 1xf64)
        arange_4 = paddle.arange(full_17, full_33, full_19, dtype="float32")
        del full_17, full_19, full_33

        # builtin.combine: ([192xf32, 192xf32]) <- (192xf32, 192xf32)
        combine_16 = [arange_4, arange_4]
        del arange_4

        # pd_op.meshgrid: ([192x192xf32, 192x192xf32]) <- ([192xf32, 192xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_16)
        del combine_16

        # builtin.split: (192x192xf32, 192x192xf32) <- ([192x192xf32, 192x192xf32])
        (
            split_9,
            split_10,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_11 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (2x300x192x192xf32) <- (192x192xf32, 2x300x192x192xf32)
        multiply_3 = paddle._C_ops.multiply(split_10, cast_11)
        del cast_11, split_10

        # pd_op.flatten: (2x300x36864xf32) <- (2x300x192x192xf32)
        flatten_7 = paddle._C_ops.flatten(multiply_3, 2, 3)

        # pd_op.max: (2x300xf32) <- (2x300x36864xf32, 1xi64)
        max_1 = paddle._C_ops.max(flatten_7, full_int_array_0, False)
        del flatten_7

        # pd_op.scale: (2x300xf32) <- (2x300xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(max_1, full_6, float("1"), True)
        del max_1

        # pd_op.full: (xf32) <- ()
        full_34 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_34,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del full_34

        # pd_op.full_like: (2x300x192x192xf32) <- (2x300x192x192xf32, 1xf32)
        full_like_9 = paddle._C_ops.full_like(
            multiply_3,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_10 = paddle._C_ops.full_like(
            assign_value__5,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (2x300x192x192xb) <- (2x300x192x192xb, 1xf32)
        full_like_11 = paddle._C_ops.full_like(
            greater_than_1,
            full_5,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_12 = paddle._C_ops.cast(full_like_11, paddle.float32)
        del full_like_11

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_13 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, xf32)
        add_22 = paddle._C_ops.add(full_like_9, full_like_10)
        del full_like_10, full_like_9

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_23 = paddle._C_ops.add(add_22, cast_12)
        del add_22, cast_12

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_24 = paddle._C_ops.add(multiply_3, add_23)
        del multiply_3

        # pd_op.add: (2x300x192x192xf32) <- (xf32, 2x300x192x192xf32)
        add_25 = paddle._C_ops.add(assign_value__5, add_23)
        del assign_value__5

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_26 = paddle._C_ops.add(cast_13, add_23)
        del add_23, cast_13

        # pd_op.cast: (2x300x192x192xb) <- (2x300x192x192xf32)
        cast_14 = paddle._C_ops.cast(add_26, paddle.bool)
        del add_26

        # pd_op.where: (2x300x192x192xf32) <- (2x300x192x192xb, 2x300x192x192xf32, 2x300x192x192xf32)
        where_2 = paddle._C_ops.where(cast_14, add_24, add_25)
        del add_24, add_25, cast_14

        # pd_op.flatten: (2x300x36864xf32) <- (2x300x192x192xf32)
        flatten_8 = paddle._C_ops.flatten(where_2, 2, 3)
        del where_2

        # pd_op.min: (2x300xf32) <- (2x300x36864xf32, 1xi64)
        min_0 = paddle._C_ops.min(flatten_8, full_int_array_0, False)
        del flatten_8

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_15 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.multiply: (2x300x192x192xf32) <- (192x192xf32, 2x300x192x192xf32)
        multiply_4 = paddle._C_ops.multiply(split_9, cast_15)
        del cast_15, split_9

        # pd_op.flatten: (2x300x36864xf32) <- (2x300x192x192xf32)
        flatten_9 = paddle._C_ops.flatten(multiply_4, 2, 3)

        # pd_op.max: (2x300xf32) <- (2x300x36864xf32, 1xi64)
        max_2 = paddle._C_ops.max(flatten_9, full_int_array_0, False)
        del flatten_9

        # pd_op.scale: (2x300xf32) <- (2x300xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(max_2, full_6, float("1"), True)
        del max_2

        # pd_op.full: (xf32) <- ()
        full_35 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_35,
            [],
            paddle.float32,
            [float("1e+08")],
            paddle.framework._current_expected_place(),
        )
        del full_35

        # pd_op.full_like: (2x300x192x192xf32) <- (2x300x192x192xf32, 1xf32)
        full_like_12 = paddle._C_ops.full_like(
            multiply_4,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (xf32) <- (xf32, 1xf32)
        full_like_13 = paddle._C_ops.full_like(
            assign_value__6,
            full_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (2x300x192x192xb) <- (2x300x192x192xb, 1xf32)
        full_like_14 = paddle._C_ops.full_like(
            greater_than_1,
            full_5,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_16 = paddle._C_ops.cast(full_like_14, paddle.float32)
        del full_like_14

        # pd_op.cast: (2x300x192x192xf32) <- (2x300x192x192xb)
        cast_17 = paddle._C_ops.cast(greater_than_1, paddle.float32)

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, xf32)
        add_27 = paddle._C_ops.add(full_like_12, full_like_13)
        del full_like_12, full_like_13

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_28 = paddle._C_ops.add(add_27, cast_16)
        del add_27, cast_16

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_29 = paddle._C_ops.add(multiply_4, add_28)
        del multiply_4

        # pd_op.add: (2x300x192x192xf32) <- (xf32, 2x300x192x192xf32)
        add_30 = paddle._C_ops.add(assign_value__6, add_28)
        del assign_value__6

        # pd_op.add: (2x300x192x192xf32) <- (2x300x192x192xf32, 2x300x192x192xf32)
        add_31 = paddle._C_ops.add(cast_17, add_28)
        del add_28, cast_17

        # pd_op.cast: (2x300x192x192xb) <- (2x300x192x192xf32)
        cast_18 = paddle._C_ops.cast(add_31, paddle.bool)
        del add_31

        # pd_op.where: (2x300x192x192xf32) <- (2x300x192x192xb, 2x300x192x192xf32, 2x300x192x192xf32)
        where_3 = paddle._C_ops.where(cast_18, add_29, add_30)
        del add_29, add_30, cast_18

        # pd_op.flatten: (2x300x36864xf32) <- (2x300x192x192xf32)
        flatten_10 = paddle._C_ops.flatten(where_3, 2, 3)
        del where_3

        # pd_op.min: (2x300xf32) <- (2x300x36864xf32, 1xi64)
        min_1 = paddle._C_ops.min(flatten_10, full_int_array_0, False)
        del flatten_10

        # builtin.combine: ([2x300xf32, 2x300xf32, 2x300xf32, 2x300xf32]) <- (2x300xf32, 2x300xf32, 2x300xf32, 2x300xf32)
        combine_17 = [min_0, min_1, scale_15, scale_16]
        del min_0, min_1, scale_15, scale_16

        # pd_op.stack: (2x300x4xf32) <- ([2x300xf32, 2x300xf32, 2x300xf32, 2x300xf32])
        stack_8 = paddle._C_ops.stack(combine_17, -1)
        del combine_17

        # pd_op.any: (2x300xb) <- (2x300x192x192xb)
        any_0 = paddle._C_ops.any(greater_than_1, [2, 3], False)
        del greater_than_1

        # pd_op.unsqueeze: (2x300x1xb) <- (2x300xb, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(any_0, full_int_array_3)
        del any_0

        # pd_op.cast: (2x300x1xf32) <- (2x300x1xb)
        cast_19 = paddle._C_ops.cast(unsqueeze_4, paddle.float32)
        del unsqueeze_4

        # pd_op.multiply: (2x300x4xf32) <- (2x300x4xf32, 2x300x1xf32)
        multiply_5 = paddle._C_ops.multiply(stack_8, cast_19)
        del cast_19, stack_8

        # pd_op.full: (4xi64) <- ()
        full_36 = paddle._C_ops.full(
            [4], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (4xi64) <- (4xi64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_36,
            [4],
            paddle.int64,
            [float("192"), float("192"), float("192"), float("192")],
            paddle.framework._current_expected_place(),
        )
        del full_36

        # pd_op.cast: (4xf32) <- (4xi64)
        cast_20 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.divide: (2x300x4xf32) <- (2x300x4xf32, 4xf32)
        divide_5 = paddle._C_ops.divide(multiply_5, cast_20)
        del cast_20, multiply_5

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32]) <- (2x300x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(divide_5, 4, full_37)
        del divide_5, full_37

        # builtin.split: (2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32) <- ([2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32])
        (
            split_11,
            split_12,
            split_13,
            split_14,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.add: (2x300x1xf32) <- (2x300x1xf32, 2x300x1xf32)
        add_32 = paddle._C_ops.add(split_11, split_13)

        # pd_op.scale: (2x300x1xf32) <- (2x300x1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(add_32, full_8, float("0"), True)
        del add_32

        # pd_op.add: (2x300x1xf32) <- (2x300x1xf32, 2x300x1xf32)
        add_33 = paddle._C_ops.add(split_12, split_14)

        # pd_op.scale: (2x300x1xf32) <- (2x300x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(add_33, full_8, float("0"), True)
        del add_33

        # pd_op.subtract: (2x300x1xf32) <- (2x300x1xf32, 2x300x1xf32)
        subtract_0 = paddle._C_ops.subtract(split_13, split_11)
        del split_11, split_13

        # pd_op.subtract: (2x300x1xf32) <- (2x300x1xf32, 2x300x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_14, split_12)
        del split_12, split_14

        # builtin.combine: ([2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32]) <- (2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32)
        combine_18 = [scale_17, scale_18, subtract_0, subtract_1]
        del scale_17, scale_18, subtract_0, subtract_1

        # pd_op.concat: (2x300x4xf32) <- ([2x300x1xf32, 2x300x1xf32, 2x300x1xf32, 2x300x1xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_18, full_9)
        del combine_18, full_9

        # pd_op.clip: (2x300x4xf32) <- (2x300x4xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(concat_8, full_5, full_6)
        del concat_8

        # pd_op.clip: (2x300x4xf32) <- (2x300x4xf32, 1xf32, 1xf32)
        clip_4 = paddle._C_ops.clip(clip_3, full_11, full_12)

        # pd_op.scale: (2x300x4xf32) <- (2x300x4xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(clip_3, full_13, float("1"), True)
        del clip_3

        # pd_op.clip: (2x300x4xf32) <- (2x300x4xf32, 1xf32, 1xf32)
        clip_5 = paddle._C_ops.clip(scale_19, full_11, full_12)
        del scale_19

        # pd_op.divide: (2x300x4xf32) <- (2x300x4xf32, 2x300x4xf32)
        divide_6 = paddle._C_ops.divide(clip_4, clip_5)
        del clip_4, clip_5

        # pd_op.log: (2x300x4xf32) <- (2x300x4xf32)
        log_2 = paddle._C_ops.log(divide_6)
        del divide_6

        # builtin.combine: ([2x100x4xf32, 2x300x4xf32]) <- (2x100x4xf32, 2x300x4xf32)
        combine_19 = [log_0, log_2]
        del log_0, log_2

        # pd_op.concat: (2x400x4xf32) <- ([2x100x4xf32, 2x300x4xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_19, full_0)
        del combine_19

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                concat_7, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_6, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_1 = paddle._C_ops.add(matmul_9, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_6, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_34 = paddle._C_ops.add(matmul_10, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_4 = paddle._C_ops.relu(add_34)
        del add_34

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_11 = paddle._C_ops.matmul(relu_4, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_35 = paddle._C_ops.add(matmul_11, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_5 = paddle._C_ops.relu(add_35)
        del add_35

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_12 = paddle._C_ops.matmul(relu_5, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_36 = paddle._C_ops.add(matmul_12, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_1 = paddle._C_ops.bmm(add_36, flatten_6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_415 = [2, 400, 192, 192]

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(bmm_1, full_int_array_415)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(concat_9)

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__1 = concat_9.detach()
        del concat_9

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(share_data__1)
        del share_data__1

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(sigmoid_2, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_13 = paddle._C_ops.matmul(sigmoid_2, parameter_123, False, False)

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_37 = paddle._C_ops.add(matmul_13, parameter_122)

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_6 = paddle._C_ops.relu(add_37)
        del add_37

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_14 = paddle._C_ops.matmul(relu_6, parameter_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_38 = paddle._C_ops.add(matmul_14, parameter_120)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_39 = paddle._C_ops.add(concat_7, add_38)

        # pd_op.full: (400x400xf32) <- ()
        full_38 = paddle._C_ops.full(
            [400, 400],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (400x400xf32) <- ()
        full_39 = paddle._C_ops.full(
            [400, 400],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_4 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_416 = [256]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_131 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_132 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_133 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_134 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_135 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_136 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_137 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_138 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_139 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_140 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_141 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_142 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_143 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_144 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_145 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_146 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_147 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_148 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_149 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_150 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_151 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_152 = full_int_array_416

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_153 = full_int_array_416

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_2, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_15 = paddle._C_ops.matmul(add_39, slice_5, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_40 = paddle._C_ops.add(matmul_15, slice_6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_417 = [0, 0, 8, 32]

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_40, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_418 = [512]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_154 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_155 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_156 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_157 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_158 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_159 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_160 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_161 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_162 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_163 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_164 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_165 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_166 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_167 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_168 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_169 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_170 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_171 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_172 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_173 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_174 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_175 = full_int_array_418

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_176 = full_int_array_418

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_16 = paddle._C_ops.matmul(add_39, slice_7, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_41 = paddle._C_ops.add(matmul_16, slice_8)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_41, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_1, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_1

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_2, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_2

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_17 = paddle._C_ops.matmul(concat_7, slice_9, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_42 = paddle._C_ops.add(matmul_17, slice_10)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_42, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_18 = paddle._C_ops.matmul(transpose_3, transpose_4, False, True)

        # pd_op.full: (1xf32) <- ()
        full_40 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_177 = full_40

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_178 = full_40

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_179 = full_40

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_180 = full_40

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_181 = full_40

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(matmul_18, full_40, float("0"), True)
        del matmul_18

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_43 = paddle._C_ops.add(scale_20, where_4)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_0 = paddle._C_ops.softmax(add_43, -1)
        del add_43

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_19 = paddle._C_ops.matmul(softmax_0, transpose_5, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_6 = paddle._C_ops.transpose(matmul_19, [0, 2, 1, 3])
        del matmul_19

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_419 = [0, 0, 256]

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_6, full_int_array_419)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_20 = paddle._C_ops.matmul(reshape_11, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_44 = paddle._C_ops.add(matmul_20, parameter_118)
        del parameter_118

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_45 = paddle._C_ops.add(concat_7, add_44)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_45, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_46 = paddle._C_ops.add(layer_norm_9, add_38)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_21 = paddle._C_ops.matmul(concat_0, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_47 = paddle._C_ops.add(matmul_21, parameter_114)
        del parameter_114

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_420 = [2, 12096, 8, 32]

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_47, full_int_array_420)

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_22 = paddle._C_ops.matmul(add_46, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_48 = paddle._C_ops.add(matmul_22, parameter_112)
        del parameter_112

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_421 = [2, 400, 8, 3, 4, 2]

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_13 = paddle._C_ops.reshape(add_48, full_int_array_421)

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_23 = paddle._C_ops.matmul(add_46, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_49 = paddle._C_ops.add(matmul_23, parameter_110)
        del parameter_110

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_422 = [2, 400, 8, 12]

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_49, full_int_array_422)

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_14, -1)
        del reshape_14

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_423 = [2, 400, 8, 3, 4]

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_15 = paddle._C_ops.reshape(softmax_1, full_int_array_423)

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            unsqueeze_5, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_424 = [2, 4]

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(slice_11, full_int_array_424)
        del slice_11

        # pd_op.full: (1xf32) <- ()
        full_41 = paddle._C_ops.full(
            [1], float("0.25"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_182 = full_41

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_183 = full_41

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_184 = full_41

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_185 = full_41

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_186 = full_41

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(reshape_13, full_41, float("0"), True)
        del reshape_13

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            unsqueeze_5, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del unsqueeze_5

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(slice_12, full_int_array_424)
        del slice_12

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_6 = paddle._C_ops.multiply(scale_21, unsqueeze_7)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(multiply_6, full_8, float("0"), True)
        del multiply_6

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_50 = paddle._C_ops.add(unsqueeze_6, scale_22)

        # pd_op.full: (3x2xi64) <- ()
        full_42 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_42,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_42

        # pd_op.full: (3xi64) <- ()
        full_43 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_43,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_43

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            slice_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            slice_13, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_13

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_7 = paddle._C_ops.multiply(slice_14, slice_15)
        del slice_14, slice_15

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            slice_16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            slice_16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_16

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_8 = paddle._C_ops.multiply(slice_17, slice_18)
        del slice_17, slice_18

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_425 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_187 = full_int_array_425

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_188 = full_int_array_425

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_189 = full_int_array_425

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_190 = full_int_array_425

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_191 = full_int_array_425

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_192 = full_int_array_425

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            slice_19, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            slice_19, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_19

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_9 = paddle._C_ops.multiply(slice_20, slice_21)
        del slice_20, slice_21

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_20 = [multiply_7, multiply_8, multiply_9]
        del multiply_7, multiply_8, multiply_9

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_15 = paddle._C_ops.split(reshape_12, stack_9, full_0)
        del reshape_12, stack_9

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_16,
            split_17,
            split_18,
        ) = split_15
        del split_15

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(add_50, full_10, float("0"), True)
        del add_50

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(scale_23, full_6, float("-1"), True)
        del scale_23

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            slice_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            slice_22, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_22

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_11 = paddle._C_ops.flatten(split_16, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_7 = paddle._C_ops.transpose(flatten_11, [0, 2, 1])
        del flatten_11

        # pd_op.full: (xi64) <- ()
        full_44 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_45 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_21 = [full_44, full_45, slice_23, slice_24]
        del slice_23, slice_24

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_7, stack_10)
        del stack_10

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            scale_24, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_8 = paddle._C_ops.transpose(slice_25, [0, 2, 1, 3, 4])
        del slice_25

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_12 = paddle._C_ops.flatten(transpose_8, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            reshape_16, flatten_12, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            slice_26, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            slice_26, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_26

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_13 = paddle._C_ops.flatten(split_17, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_9 = paddle._C_ops.transpose(flatten_13, [0, 2, 1])
        del flatten_13

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_22 = [full_44, full_45, slice_27, slice_28]
        del slice_27, slice_28

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_9, stack_11)
        del stack_11

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            scale_24, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_10 = paddle._C_ops.transpose(slice_29, [0, 2, 1, 3, 4])
        del slice_29

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_14 = paddle._C_ops.flatten(transpose_10, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            reshape_17, flatten_14, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            assign_value__8, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__8

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            slice_30, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            slice_30, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_30

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_15 = paddle._C_ops.flatten(split_18, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_11 = paddle._C_ops.transpose(flatten_15, [0, 2, 1])
        del flatten_15

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_23 = [full_44, full_45, slice_31, slice_32]
        del slice_31, slice_32

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_11, stack_12)
        del stack_12

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            scale_24, [3], full_int_array_3, full_int_array_425, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_12 = paddle._C_ops.transpose(slice_33, [0, 2, 1, 3, 4])
        del slice_33

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_16 = paddle._C_ops.flatten(transpose_12, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            reshape_18, flatten_16, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3, 4])
        del reshape_15

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_426 = [16, 1, 400, 12]

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(transpose_13, full_int_array_426)

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_24 = [grid_sample_0, grid_sample_1, grid_sample_2]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_13 = paddle._C_ops.stack(combine_24, -2)
        del combine_24

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_17 = paddle._C_ops.flatten(stack_13, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_10 = paddle._C_ops.multiply(flatten_17, reshape_19)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_10, full_int_array_0, None, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_427 = [2, 256, 400]

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_20 = paddle._C_ops.reshape(sum_0, full_int_array_427)

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_20, [0, 2, 1])
        del reshape_20

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_24 = paddle._C_ops.matmul(transpose_14, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_51 = paddle._C_ops.add(matmul_24, parameter_108)
        del parameter_108

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_52 = paddle._C_ops.add(layer_norm_9, add_51)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_107, parameter_106, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_106, parameter_107

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_12, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_53 = paddle._C_ops.add(matmul_25, parameter_104)
        del parameter_104

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_7 = paddle._C_ops.relu(add_53)
        del add_53

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_26 = paddle._C_ops.matmul(relu_7, parameter_103, False, False)
        del parameter_103

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_54 = paddle._C_ops.add(matmul_26, parameter_102)
        del parameter_102

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_55 = paddle._C_ops.add(layer_norm_12, add_54)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_55, parameter_101, parameter_100, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_100, parameter_101

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_27 = paddle._C_ops.matmul(layer_norm_15, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_56 = paddle._C_ops.add(matmul_27, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_8 = paddle._C_ops.relu(add_56)
        del add_56

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_28 = paddle._C_ops.matmul(relu_8, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_57 = paddle._C_ops.add(matmul_28, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_9 = paddle._C_ops.relu(add_57)
        del add_57

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_29 = paddle._C_ops.matmul(relu_9, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_58 = paddle._C_ops.add(matmul_29, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_6 = paddle._C_ops.clip(sigmoid_2, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_7 = paddle._C_ops.clip(clip_6, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(clip_6, full_13, float("1"), True)
        del clip_6

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_8 = paddle._C_ops.clip(scale_25, full_11, full_12)
        del scale_25

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_7 = paddle._C_ops.divide(clip_7, clip_8)
        del clip_7, clip_8

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_3 = paddle._C_ops.log(divide_7)
        del divide_7

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_59 = paddle._C_ops.add(add_58, log_3)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_59)
        del add_59

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_15, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_18, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_60 = paddle._C_ops.add(matmul_30, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_18, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_61 = paddle._C_ops.add(matmul_31, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_10 = paddle._C_ops.relu(add_61)
        del add_61

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_32 = paddle._C_ops.matmul(relu_10, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_62 = paddle._C_ops.add(matmul_32, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_11 = paddle._C_ops.relu(add_62)
        del add_62

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_33 = paddle._C_ops.matmul(relu_11, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_63 = paddle._C_ops.add(matmul_33, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_2 = paddle._C_ops.bmm(add_63, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(bmm_2, full_int_array_415)

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__2 = sigmoid_3.detach()

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(share_data__2, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_34 = paddle._C_ops.matmul(share_data__2, parameter_123, False, False)

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_64 = paddle._C_ops.add(matmul_34, parameter_122)

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_12 = paddle._C_ops.relu(add_64)
        del add_64

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_35 = paddle._C_ops.matmul(relu_12, parameter_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_65 = paddle._C_ops.add(matmul_35, parameter_120)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_66 = paddle._C_ops.add(layer_norm_15, add_65)

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_5 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            data_3, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            data_4, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_36 = paddle._C_ops.matmul(add_66, slice_34, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_67 = paddle._C_ops.add(matmul_36, slice_35)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_67, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            data_3, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            data_4, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_37 = paddle._C_ops.matmul(add_66, slice_36, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_68 = paddle._C_ops.add(matmul_37, slice_37)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_68, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_23, [0, 2, 1, 3])
        del reshape_23

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            data_3, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_3

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            data_4, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_4

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_15, slice_38, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_69 = paddle._C_ops.add(matmul_38, slice_39)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_69, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_39 = paddle._C_ops.matmul(transpose_15, transpose_16, False, True)

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(matmul_39, full_40, float("0"), True)
        del matmul_39

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_70 = paddle._C_ops.add(scale_26, where_5)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_2 = paddle._C_ops.softmax(add_70, -1)
        del add_70

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_40 = paddle._C_ops.matmul(softmax_2, transpose_17, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_18, full_int_array_419)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_25, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_71 = paddle._C_ops.add(matmul_41, parameter_98)
        del parameter_98

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_72 = paddle._C_ops.add(layer_norm_15, add_71)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_72, parameter_97, parameter_96, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_96, parameter_97

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_73 = paddle._C_ops.add(layer_norm_21, add_65)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_42 = paddle._C_ops.matmul(concat_0, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_74 = paddle._C_ops.add(matmul_42, parameter_94)
        del parameter_94

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_74, full_int_array_420)

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_43 = paddle._C_ops.matmul(add_73, parameter_93, False, False)
        del parameter_93

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_75 = paddle._C_ops.add(matmul_43, parameter_92)
        del parameter_92

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_27 = paddle._C_ops.reshape(add_75, full_int_array_421)

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_44 = paddle._C_ops.matmul(add_73, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_76 = paddle._C_ops.add(matmul_44, parameter_90)
        del parameter_90

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_76, full_int_array_422)

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_28, -1)
        del reshape_28

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_29 = paddle._C_ops.reshape(softmax_3, full_int_array_423)

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            unsqueeze_8, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(slice_40, full_int_array_424)
        del slice_40

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(reshape_27, full_41, float("0"), True)
        del reshape_27

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            unsqueeze_8, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del unsqueeze_8

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(slice_41, full_int_array_424)
        del slice_41

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_11 = paddle._C_ops.multiply(scale_27, unsqueeze_10)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(multiply_11, full_8, float("0"), True)
        del multiply_11

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_77 = paddle._C_ops.add(unsqueeze_9, scale_28)

        # pd_op.full: (3x2xi64) <- ()
        full_46 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_46,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_46

        # pd_op.full: (3xi64) <- ()
        full_47 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_47,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_47

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            slice_42, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            slice_42, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_42

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_12 = paddle._C_ops.multiply(slice_43, slice_44)
        del slice_43, slice_44

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            slice_45, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            slice_45, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_45

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_13 = paddle._C_ops.multiply(slice_46, slice_47)
        del slice_46, slice_47

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            slice_48, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            slice_48, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_48

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_14 = paddle._C_ops.multiply(slice_49, slice_50)
        del slice_49, slice_50

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_25 = [multiply_12, multiply_13, multiply_14]
        del multiply_12, multiply_13, multiply_14

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_19 = paddle._C_ops.split(reshape_26, stack_14, full_0)
        del reshape_26, stack_14

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_20,
            split_21,
            split_22,
        ) = split_19
        del split_19

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(add_77, full_10, float("0"), True)
        del add_77

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(scale_29, full_6, float("-1"), True)
        del scale_29

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            slice_51, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            slice_51, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_51

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_18 = paddle._C_ops.flatten(split_20, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_19 = paddle._C_ops.transpose(flatten_18, [0, 2, 1])
        del flatten_18

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_26 = [full_44, full_45, slice_52, slice_53]
        del slice_52, slice_53

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(transpose_19, stack_15)
        del stack_15

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            scale_30, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_20 = paddle._C_ops.transpose(slice_54, [0, 2, 1, 3, 4])
        del slice_54

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_19 = paddle._C_ops.flatten(transpose_20, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_3 = paddle._C_ops.grid_sample(
            reshape_30, flatten_19, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            slice_55, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            slice_55, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_55

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_20 = paddle._C_ops.flatten(split_21, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_21 = paddle._C_ops.transpose(flatten_20, [0, 2, 1])
        del flatten_20

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_27 = [full_44, full_45, slice_56, slice_57]
        del slice_56, slice_57

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(transpose_21, stack_16)
        del stack_16

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            scale_30, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_22 = paddle._C_ops.transpose(slice_58, [0, 2, 1, 3, 4])
        del slice_58

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_21 = paddle._C_ops.flatten(transpose_22, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_4 = paddle._C_ops.grid_sample(
            reshape_31, flatten_21, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            assign_value__10, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__10

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            slice_59, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            slice_59, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_59

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_22 = paddle._C_ops.flatten(split_22, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_23 = paddle._C_ops.transpose(flatten_22, [0, 2, 1])
        del flatten_22

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_28 = [full_44, full_45, slice_60, slice_61]
        del slice_60, slice_61

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(transpose_23, stack_17)
        del stack_17

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            scale_30, [3], full_int_array_3, full_int_array_425, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_24 = paddle._C_ops.transpose(slice_62, [0, 2, 1, 3, 4])
        del slice_62

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_23 = paddle._C_ops.flatten(transpose_24, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_5 = paddle._C_ops.grid_sample(
            reshape_32, flatten_23, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_25 = paddle._C_ops.transpose(reshape_29, [0, 2, 1, 3, 4])
        del reshape_29

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_25, full_int_array_426)

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_29 = [grid_sample_3, grid_sample_4, grid_sample_5]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_18 = paddle._C_ops.stack(combine_29, -2)
        del combine_29

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_24 = paddle._C_ops.flatten(stack_18, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_15 = paddle._C_ops.multiply(flatten_24, reshape_33)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_15, full_int_array_0, None, False)

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_34 = paddle._C_ops.reshape(sum_1, full_int_array_427)

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_34, [0, 2, 1])
        del reshape_34

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_45 = paddle._C_ops.matmul(transpose_26, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_78 = paddle._C_ops.add(matmul_45, parameter_88)
        del parameter_88

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_79 = paddle._C_ops.add(layer_norm_21, add_78)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_79, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_86, parameter_87

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_24, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_80 = paddle._C_ops.add(matmul_46, parameter_84)
        del parameter_84

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_13 = paddle._C_ops.relu(add_80)
        del add_80

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_47 = paddle._C_ops.matmul(relu_13, parameter_83, False, False)
        del parameter_83

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_81 = paddle._C_ops.add(matmul_47, parameter_82)
        del parameter_82

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_82 = paddle._C_ops.add(layer_norm_24, add_81)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_82, parameter_81, parameter_80, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_80, parameter_81

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_27, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_83 = paddle._C_ops.add(matmul_48, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_14 = paddle._C_ops.relu(add_83)
        del add_83

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_49 = paddle._C_ops.matmul(relu_14, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_84 = paddle._C_ops.add(matmul_49, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_15 = paddle._C_ops.relu(add_84)
        del add_84

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_50 = paddle._C_ops.matmul(relu_15, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_85 = paddle._C_ops.add(matmul_50, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_9 = paddle._C_ops.clip(share_data__2, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_10 = paddle._C_ops.clip(clip_9, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(clip_9, full_13, float("1"), True)
        del clip_9

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_11 = paddle._C_ops.clip(scale_31, full_11, full_12)
        del scale_31

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_8 = paddle._C_ops.divide(clip_10, clip_11)
        del clip_10, clip_11

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_4 = paddle._C_ops.log(divide_8)
        del divide_8

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_86 = paddle._C_ops.add(add_85, log_4)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_86)
        del add_86

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_27, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_30, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_87 = paddle._C_ops.add(matmul_51, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_30, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_88 = paddle._C_ops.add(matmul_52, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_16 = paddle._C_ops.relu(add_88)
        del add_88

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_53 = paddle._C_ops.matmul(relu_16, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_89 = paddle._C_ops.add(matmul_53, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_17 = paddle._C_ops.relu(add_89)
        del add_89

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_54 = paddle._C_ops.matmul(relu_17, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_90 = paddle._C_ops.add(matmul_54, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_3 = paddle._C_ops.bmm(add_90, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(bmm_3, full_int_array_415)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_27, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_91 = paddle._C_ops.add(matmul_55, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_18 = paddle._C_ops.relu(add_91)
        del add_91

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_56 = paddle._C_ops.matmul(relu_18, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_92 = paddle._C_ops.add(matmul_56, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_19 = paddle._C_ops.relu(add_92)
        del add_92

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_57 = paddle._C_ops.matmul(relu_19, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_93 = paddle._C_ops.add(matmul_57, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_12 = paddle._C_ops.clip(sigmoid_3, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_13 = paddle._C_ops.clip(clip_12, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(clip_12, full_13, float("1"), True)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_14 = paddle._C_ops.clip(scale_32, full_11, full_12)

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_9 = paddle._C_ops.divide(clip_13, clip_14)

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_5 = paddle._C_ops.log(divide_9)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_94 = paddle._C_ops.add(add_93, log_5)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(add_94)
        del add_94

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__3 = sigmoid_4.detach()

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(share_data__3, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_58 = paddle._C_ops.matmul(share_data__3, parameter_123, False, False)

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_95 = paddle._C_ops.add(matmul_58, parameter_122)

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_20 = paddle._C_ops.relu(add_95)
        del add_95

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_59 = paddle._C_ops.matmul(relu_20, parameter_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_96 = paddle._C_ops.add(matmul_59, parameter_120)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_97 = paddle._C_ops.add(layer_norm_27, add_96)

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_6 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            data_5, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            data_6, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_60 = paddle._C_ops.matmul(add_97, slice_63, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_98 = paddle._C_ops.add(matmul_60, slice_64)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_98, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            data_5, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            data_6, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_61 = paddle._C_ops.matmul(add_97, slice_65, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_99 = paddle._C_ops.add(matmul_61, slice_66)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(add_99, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            data_5, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_5

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            data_6, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_6

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_27, slice_67, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_100 = paddle._C_ops.add(matmul_62, slice_68)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_100, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_63 = paddle._C_ops.matmul(transpose_27, transpose_28, False, True)

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(matmul_63, full_40, float("0"), True)
        del matmul_63

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_101 = paddle._C_ops.add(scale_33, where_6)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_4 = paddle._C_ops.softmax(add_101, -1)
        del add_101

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_64 = paddle._C_ops.matmul(softmax_4, transpose_29, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])
        del matmul_64

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_30, full_int_array_419)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_65 = paddle._C_ops.matmul(reshape_39, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_102 = paddle._C_ops.add(matmul_65, parameter_78)
        del parameter_78

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_103 = paddle._C_ops.add(layer_norm_27, add_102)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_103, parameter_77, parameter_76, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_76, parameter_77

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_104 = paddle._C_ops.add(layer_norm_33, add_96)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_66 = paddle._C_ops.matmul(concat_0, parameter_75, False, False)
        del parameter_75

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_105 = paddle._C_ops.add(matmul_66, parameter_74)
        del parameter_74

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_105, full_int_array_420)

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_67 = paddle._C_ops.matmul(add_104, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_106 = paddle._C_ops.add(matmul_67, parameter_72)
        del parameter_72

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_41 = paddle._C_ops.reshape(add_106, full_int_array_421)

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_68 = paddle._C_ops.matmul(add_104, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_107 = paddle._C_ops.add(matmul_68, parameter_70)
        del parameter_70

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_107, full_int_array_422)

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_42, -1)
        del reshape_42

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_43 = paddle._C_ops.reshape(softmax_5, full_int_array_423)

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            unsqueeze_11, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(slice_69, full_int_array_424)
        del slice_69

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(reshape_41, full_41, float("0"), True)
        del reshape_41

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            unsqueeze_11, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del unsqueeze_11

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(slice_70, full_int_array_424)
        del slice_70

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_16 = paddle._C_ops.multiply(scale_34, unsqueeze_13)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(multiply_16, full_8, float("0"), True)
        del multiply_16

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_108 = paddle._C_ops.add(unsqueeze_12, scale_35)

        # pd_op.full: (3x2xi64) <- ()
        full_48 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_48,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_48

        # pd_op.full: (3xi64) <- ()
        full_49 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_49,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_49

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            slice_71, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            slice_71, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_71

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_17 = paddle._C_ops.multiply(slice_72, slice_73)
        del slice_72, slice_73

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            slice_74, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            slice_74, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_74

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_18 = paddle._C_ops.multiply(slice_75, slice_76)
        del slice_75, slice_76

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            slice_77, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            slice_77, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_77

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_19 = paddle._C_ops.multiply(slice_78, slice_79)
        del slice_78, slice_79

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_30 = [multiply_17, multiply_18, multiply_19]
        del multiply_17, multiply_18, multiply_19

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_30, 0)
        del combine_30

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_23 = paddle._C_ops.split(reshape_40, stack_19, full_0)
        del reshape_40, stack_19

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_24,
            split_25,
            split_26,
        ) = split_23
        del split_23

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(add_108, full_10, float("0"), True)
        del add_108

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(scale_36, full_6, float("-1"), True)
        del scale_36

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            slice_80, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            slice_80, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_80

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_25 = paddle._C_ops.flatten(split_24, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_31 = paddle._C_ops.transpose(flatten_25, [0, 2, 1])
        del flatten_25

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_31 = [full_44, full_45, slice_81, slice_82]
        del slice_81, slice_82

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_31, 0)
        del combine_31

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(transpose_31, stack_20)
        del stack_20

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            scale_37, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_32 = paddle._C_ops.transpose(slice_83, [0, 2, 1, 3, 4])
        del slice_83

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_26 = paddle._C_ops.flatten(transpose_32, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_6 = paddle._C_ops.grid_sample(
            reshape_44, flatten_26, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            slice_84, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            slice_84, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_84

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_27 = paddle._C_ops.flatten(split_25, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_33 = paddle._C_ops.transpose(flatten_27, [0, 2, 1])
        del flatten_27

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_32 = [full_44, full_45, slice_85, slice_86]
        del slice_85, slice_86

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_33, stack_21)
        del stack_21

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            scale_37, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_34 = paddle._C_ops.transpose(slice_87, [0, 2, 1, 3, 4])
        del slice_87

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_28 = paddle._C_ops.flatten(transpose_34, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_7 = paddle._C_ops.grid_sample(
            reshape_45, flatten_28, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(
            assign_value__12, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__12

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(
            slice_88, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(
            slice_88, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_88

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_29 = paddle._C_ops.flatten(split_26, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_35 = paddle._C_ops.transpose(flatten_29, [0, 2, 1])
        del flatten_29

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_33 = [full_44, full_45, slice_89, slice_90]
        del slice_89, slice_90

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_33, 0)
        del combine_33

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(transpose_35, stack_22)
        del stack_22

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(
            scale_37, [3], full_int_array_3, full_int_array_425, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_36 = paddle._C_ops.transpose(slice_91, [0, 2, 1, 3, 4])
        del slice_91

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_30 = paddle._C_ops.flatten(transpose_36, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_8 = paddle._C_ops.grid_sample(
            reshape_46, flatten_30, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_43, [0, 2, 1, 3, 4])
        del reshape_43

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_37, full_int_array_426)

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_34 = [grid_sample_6, grid_sample_7, grid_sample_8]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_23 = paddle._C_ops.stack(combine_34, -2)
        del combine_34

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_31 = paddle._C_ops.flatten(stack_23, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_20 = paddle._C_ops.multiply(flatten_31, reshape_47)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(multiply_20, full_int_array_0, None, False)

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(sum_2, full_int_array_427)

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_38 = paddle._C_ops.transpose(reshape_48, [0, 2, 1])
        del reshape_48

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_69 = paddle._C_ops.matmul(transpose_38, parameter_69, False, False)
        del parameter_69

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_109 = paddle._C_ops.add(matmul_69, parameter_68)
        del parameter_68

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_110 = paddle._C_ops.add(layer_norm_33, add_109)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_110, parameter_67, parameter_66, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_36, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_111 = paddle._C_ops.add(matmul_70, parameter_64)
        del parameter_64

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_21 = paddle._C_ops.relu(add_111)
        del add_111

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_71 = paddle._C_ops.matmul(relu_21, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_112 = paddle._C_ops.add(matmul_71, parameter_62)
        del parameter_62

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_113 = paddle._C_ops.add(layer_norm_36, add_112)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_113, parameter_61, parameter_60, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_60, parameter_61

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_72 = paddle._C_ops.matmul(layer_norm_39, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_114 = paddle._C_ops.add(matmul_72, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_22 = paddle._C_ops.relu(add_114)
        del add_114

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_73 = paddle._C_ops.matmul(relu_22, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_115 = paddle._C_ops.add(matmul_73, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_23 = paddle._C_ops.relu(add_115)
        del add_115

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_74 = paddle._C_ops.matmul(relu_23, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_116 = paddle._C_ops.add(matmul_74, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_15 = paddle._C_ops.clip(share_data__3, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_16 = paddle._C_ops.clip(clip_15, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(clip_15, full_13, float("1"), True)
        del clip_15

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_17 = paddle._C_ops.clip(scale_38, full_11, full_12)
        del scale_38

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_10 = paddle._C_ops.divide(clip_16, clip_17)
        del clip_16, clip_17

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_6 = paddle._C_ops.log(divide_10)
        del divide_10

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_117 = paddle._C_ops.add(add_116, log_6)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(add_117)
        del add_117

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_39, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_75 = paddle._C_ops.matmul(layer_norm_42, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_118 = paddle._C_ops.add(matmul_75, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_76 = paddle._C_ops.matmul(layer_norm_42, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_119 = paddle._C_ops.add(matmul_76, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_24 = paddle._C_ops.relu(add_119)
        del add_119

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_77 = paddle._C_ops.matmul(relu_24, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_120 = paddle._C_ops.add(matmul_77, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_25 = paddle._C_ops.relu(add_120)
        del add_120

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_78 = paddle._C_ops.matmul(relu_25, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_121 = paddle._C_ops.add(matmul_78, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_4 = paddle._C_ops.bmm(add_121, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(bmm_4, full_int_array_415)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_79 = paddle._C_ops.matmul(layer_norm_39, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_122 = paddle._C_ops.add(matmul_79, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_26 = paddle._C_ops.relu(add_122)
        del add_122

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_80 = paddle._C_ops.matmul(relu_26, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_123 = paddle._C_ops.add(matmul_80, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_27 = paddle._C_ops.relu(add_123)
        del add_123

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_81 = paddle._C_ops.matmul(relu_27, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_124 = paddle._C_ops.add(matmul_81, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_18 = paddle._C_ops.clip(sigmoid_4, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_19 = paddle._C_ops.clip(clip_18, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(clip_18, full_13, float("1"), True)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_20 = paddle._C_ops.clip(scale_39, full_11, full_12)

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_11 = paddle._C_ops.divide(clip_19, clip_20)

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_7 = paddle._C_ops.log(divide_11)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_125 = paddle._C_ops.add(add_124, log_7)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(add_125)
        del add_125

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__4 = sigmoid_6.detach()

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(share_data__4, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_82 = paddle._C_ops.matmul(share_data__4, parameter_123, False, False)

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_126 = paddle._C_ops.add(matmul_82, parameter_122)

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_28 = paddle._C_ops.relu(add_126)
        del add_126

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_83 = paddle._C_ops.matmul(relu_28, parameter_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_127 = paddle._C_ops.add(matmul_83, parameter_120)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_128 = paddle._C_ops.add(layer_norm_39, add_127)

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_7 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(
            data_7, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(
            data_8, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_84 = paddle._C_ops.matmul(add_128, slice_92, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_129 = paddle._C_ops.add(matmul_84, slice_93)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(add_129, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(
            data_7, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(
            data_8, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_85 = paddle._C_ops.matmul(add_128, slice_94, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_130 = paddle._C_ops.add(matmul_85, slice_95)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(add_130, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_51, [0, 2, 1, 3])
        del reshape_51

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(
            data_7, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_7

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(
            data_8, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_8

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_86 = paddle._C_ops.matmul(layer_norm_39, slice_96, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_131 = paddle._C_ops.add(matmul_86, slice_97)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_131, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_87 = paddle._C_ops.matmul(transpose_39, transpose_40, False, True)

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(matmul_87, full_40, float("0"), True)
        del matmul_87

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_132 = paddle._C_ops.add(scale_40, where_7)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_6 = paddle._C_ops.softmax(add_132, -1)
        del add_132

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_88 = paddle._C_ops.matmul(softmax_6, transpose_41, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_42 = paddle._C_ops.transpose(matmul_88, [0, 2, 1, 3])
        del matmul_88

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_53 = paddle._C_ops.reshape(transpose_42, full_int_array_419)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_89 = paddle._C_ops.matmul(reshape_53, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_133 = paddle._C_ops.add(matmul_89, parameter_58)
        del parameter_58

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_134 = paddle._C_ops.add(layer_norm_39, add_133)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_134, parameter_57, parameter_56, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_56, parameter_57

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_135 = paddle._C_ops.add(layer_norm_45, add_127)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_90 = paddle._C_ops.matmul(concat_0, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_136 = paddle._C_ops.add(matmul_90, parameter_54)
        del parameter_54

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(add_136, full_int_array_420)

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_91 = paddle._C_ops.matmul(add_135, parameter_53, False, False)
        del parameter_53

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_137 = paddle._C_ops.add(matmul_91, parameter_52)
        del parameter_52

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_55 = paddle._C_ops.reshape(add_137, full_int_array_421)

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_92 = paddle._C_ops.matmul(add_135, parameter_51, False, False)
        del parameter_51

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_138 = paddle._C_ops.add(matmul_92, parameter_50)
        del parameter_50

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_138, full_int_array_422)

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_56, -1)
        del reshape_56

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_57 = paddle._C_ops.reshape(softmax_7, full_int_array_423)

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(
            unsqueeze_14, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(slice_98, full_int_array_424)
        del slice_98

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(reshape_55, full_41, float("0"), True)
        del reshape_55

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(
            unsqueeze_14, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del unsqueeze_14

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(slice_99, full_int_array_424)
        del slice_99

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_21 = paddle._C_ops.multiply(scale_41, unsqueeze_16)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(multiply_21, full_8, float("0"), True)
        del multiply_21

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_139 = paddle._C_ops.add(unsqueeze_15, scale_42)

        # pd_op.full: (3x2xi64) <- ()
        full_50 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__14 = paddle._C_ops.assign_value_(
            full_50,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_50

        # pd_op.full: (3xi64) <- ()
        full_51 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__15 = paddle._C_ops.assign_value_(
            full_51,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_51

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(
            slice_100, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(
            slice_100, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_100

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_22 = paddle._C_ops.multiply(slice_101, slice_102)
        del slice_101, slice_102

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(
            slice_103, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(
            slice_103, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_103

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_23 = paddle._C_ops.multiply(slice_104, slice_105)
        del slice_104, slice_105

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(
            slice_106, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(
            slice_106, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_106

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_24 = paddle._C_ops.multiply(slice_107, slice_108)
        del slice_107, slice_108

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_35 = [multiply_22, multiply_23, multiply_24]
        del multiply_22, multiply_23, multiply_24

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_35, 0)
        del combine_35

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_27 = paddle._C_ops.split(reshape_54, stack_24, full_0)
        del reshape_54, stack_24

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_28,
            split_29,
            split_30,
        ) = split_27
        del split_27

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(add_139, full_10, float("0"), True)
        del add_139

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(scale_43, full_6, float("-1"), True)
        del scale_43

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(
            slice_109, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(
            slice_109, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_109

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_32 = paddle._C_ops.flatten(split_28, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_43 = paddle._C_ops.transpose(flatten_32, [0, 2, 1])
        del flatten_32

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_36 = [full_44, full_45, slice_110, slice_111]
        del slice_110, slice_111

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_36, 0)
        del combine_36

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(transpose_43, stack_25)
        del stack_25

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(
            scale_44, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_44 = paddle._C_ops.transpose(slice_112, [0, 2, 1, 3, 4])
        del slice_112

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_33 = paddle._C_ops.flatten(transpose_44, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_9 = paddle._C_ops.grid_sample(
            reshape_58, flatten_33, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(
            slice_113, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(
            slice_113, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_113

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_34 = paddle._C_ops.flatten(split_29, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_45 = paddle._C_ops.transpose(flatten_34, [0, 2, 1])
        del flatten_34

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_37 = [full_44, full_45, slice_114, slice_115]
        del slice_114, slice_115

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_37, 0)
        del combine_37

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_59 = paddle._C_ops.reshape(transpose_45, stack_26)
        del stack_26

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(
            scale_44, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_46 = paddle._C_ops.transpose(slice_116, [0, 2, 1, 3, 4])
        del slice_116

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_35 = paddle._C_ops.flatten(transpose_46, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_10 = paddle._C_ops.grid_sample(
            reshape_59, flatten_35, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(
            assign_value__14, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__14

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(
            slice_117, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(
            slice_117, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_117

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_36 = paddle._C_ops.flatten(split_30, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_47 = paddle._C_ops.transpose(flatten_36, [0, 2, 1])
        del flatten_36

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_38 = [full_44, full_45, slice_118, slice_119]
        del slice_118, slice_119

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_38, 0)
        del combine_38

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(transpose_47, stack_27)
        del stack_27

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(
            scale_44, [3], full_int_array_3, full_int_array_425, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_48 = paddle._C_ops.transpose(slice_120, [0, 2, 1, 3, 4])
        del slice_120

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_37 = paddle._C_ops.flatten(transpose_48, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_11 = paddle._C_ops.grid_sample(
            reshape_60, flatten_37, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3, 4])
        del reshape_57

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(transpose_49, full_int_array_426)

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_39 = [grid_sample_9, grid_sample_10, grid_sample_11]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_28 = paddle._C_ops.stack(combine_39, -2)
        del combine_39

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_38 = paddle._C_ops.flatten(stack_28, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_25 = paddle._C_ops.multiply(flatten_38, reshape_61)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(multiply_25, full_int_array_0, None, False)

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_62 = paddle._C_ops.reshape(sum_3, full_int_array_427)

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_62, [0, 2, 1])
        del reshape_62

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_93 = paddle._C_ops.matmul(transpose_50, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_140 = paddle._C_ops.add(matmul_93, parameter_48)
        del parameter_48

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_141 = paddle._C_ops.add(layer_norm_45, add_140)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_141, parameter_47, parameter_46, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_46, parameter_47

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_94 = paddle._C_ops.matmul(layer_norm_48, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_142 = paddle._C_ops.add(matmul_94, parameter_44)
        del parameter_44

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_29 = paddle._C_ops.relu(add_142)
        del add_142

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_95 = paddle._C_ops.matmul(relu_29, parameter_43, False, False)
        del parameter_43

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_143 = paddle._C_ops.add(matmul_95, parameter_42)
        del parameter_42

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_144 = paddle._C_ops.add(layer_norm_48, add_143)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_144, parameter_41, parameter_40, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_40, parameter_41

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_96 = paddle._C_ops.matmul(layer_norm_51, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_145 = paddle._C_ops.add(matmul_96, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_30 = paddle._C_ops.relu(add_145)
        del add_145

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_97 = paddle._C_ops.matmul(relu_30, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_146 = paddle._C_ops.add(matmul_97, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_31 = paddle._C_ops.relu(add_146)
        del add_146

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_98 = paddle._C_ops.matmul(relu_31, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_147 = paddle._C_ops.add(matmul_98, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_21 = paddle._C_ops.clip(share_data__4, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_22 = paddle._C_ops.clip(clip_21, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(clip_21, full_13, float("1"), True)
        del clip_21

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_23 = paddle._C_ops.clip(scale_45, full_11, full_12)
        del scale_45

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_12 = paddle._C_ops.divide(clip_22, clip_23)
        del clip_22, clip_23

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_8 = paddle._C_ops.log(divide_12)
        del divide_12

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_148 = paddle._C_ops.add(add_147, log_8)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(add_148)
        del add_148

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_51, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_99 = paddle._C_ops.matmul(layer_norm_54, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_149 = paddle._C_ops.add(matmul_99, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_54, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_150 = paddle._C_ops.add(matmul_100, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_32 = paddle._C_ops.relu(add_150)
        del add_150

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_101 = paddle._C_ops.matmul(relu_32, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_151 = paddle._C_ops.add(matmul_101, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_33 = paddle._C_ops.relu(add_151)
        del add_151

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_102 = paddle._C_ops.matmul(relu_33, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_152 = paddle._C_ops.add(matmul_102, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_5 = paddle._C_ops.bmm(add_152, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(bmm_5, full_int_array_415)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_103 = paddle._C_ops.matmul(layer_norm_51, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_153 = paddle._C_ops.add(matmul_103, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_34 = paddle._C_ops.relu(add_153)
        del add_153

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_104 = paddle._C_ops.matmul(relu_34, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_154 = paddle._C_ops.add(matmul_104, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_35 = paddle._C_ops.relu(add_154)
        del add_154

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_105 = paddle._C_ops.matmul(relu_35, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_155 = paddle._C_ops.add(matmul_105, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_24 = paddle._C_ops.clip(sigmoid_6, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_25 = paddle._C_ops.clip(clip_24, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(clip_24, full_13, float("1"), True)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_26 = paddle._C_ops.clip(scale_46, full_11, full_12)

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_13 = paddle._C_ops.divide(clip_25, clip_26)

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_9 = paddle._C_ops.log(divide_13)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_156 = paddle._C_ops.add(add_155, log_9)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(add_156)
        del add_156

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__5 = sigmoid_8.detach()

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(share_data__5, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_106 = paddle._C_ops.matmul(share_data__5, parameter_123, False, False)

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_157 = paddle._C_ops.add(matmul_106, parameter_122)

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_36 = paddle._C_ops.relu(add_157)
        del add_157

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_107 = paddle._C_ops.matmul(relu_36, parameter_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_158 = paddle._C_ops.add(matmul_107, parameter_120)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_159 = paddle._C_ops.add(layer_norm_51, add_158)

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_8 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(
            data_9, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(
            data_10, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_108 = paddle._C_ops.matmul(add_159, slice_121, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_160 = paddle._C_ops.add(matmul_108, slice_122)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_160, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(
            data_9, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(
            data_10, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_109 = paddle._C_ops.matmul(add_159, slice_123, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_161 = paddle._C_ops.add(matmul_109, slice_124)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(add_161, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_65, [0, 2, 1, 3])
        del reshape_65

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(
            data_9, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_9

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(
            data_10, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_10

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_110 = paddle._C_ops.matmul(layer_norm_51, slice_125, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_162 = paddle._C_ops.add(matmul_110, slice_126)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(add_162, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_111 = paddle._C_ops.matmul(transpose_51, transpose_52, False, True)

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(matmul_111, full_40, float("0"), True)
        del matmul_111

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_163 = paddle._C_ops.add(scale_47, where_8)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_8 = paddle._C_ops.softmax(add_163, -1)
        del add_163

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_112 = paddle._C_ops.matmul(softmax_8, transpose_53, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_54 = paddle._C_ops.transpose(matmul_112, [0, 2, 1, 3])
        del matmul_112

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_54, full_int_array_419)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_113 = paddle._C_ops.matmul(reshape_67, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_164 = paddle._C_ops.add(matmul_113, parameter_38)
        del parameter_38

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_165 = paddle._C_ops.add(layer_norm_51, add_164)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_165, parameter_37, parameter_36, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_36, parameter_37

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_166 = paddle._C_ops.add(layer_norm_57, add_158)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_114 = paddle._C_ops.matmul(concat_0, parameter_35, False, False)
        del parameter_35

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_167 = paddle._C_ops.add(matmul_114, parameter_34)
        del parameter_34

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(add_167, full_int_array_420)

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_115 = paddle._C_ops.matmul(add_166, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_168 = paddle._C_ops.add(matmul_115, parameter_32)
        del parameter_32

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_69 = paddle._C_ops.reshape(add_168, full_int_array_421)

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_116 = paddle._C_ops.matmul(add_166, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_169 = paddle._C_ops.add(matmul_116, parameter_30)
        del parameter_30

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(add_169, full_int_array_422)

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_70, -1)
        del reshape_70

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_71 = paddle._C_ops.reshape(softmax_9, full_int_array_423)

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(
            unsqueeze_17, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(slice_127, full_int_array_424)
        del slice_127

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(reshape_69, full_41, float("0"), True)
        del reshape_69

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(
            unsqueeze_17, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del unsqueeze_17

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(slice_128, full_int_array_424)
        del slice_128

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_26 = paddle._C_ops.multiply(scale_48, unsqueeze_19)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(multiply_26, full_8, float("0"), True)
        del multiply_26

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_170 = paddle._C_ops.add(unsqueeze_18, scale_49)

        # pd_op.full: (3x2xi64) <- ()
        full_52 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__16 = paddle._C_ops.assign_value_(
            full_52,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_52

        # pd_op.full: (3xi64) <- ()
        full_53 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__17 = paddle._C_ops.assign_value_(
            full_53,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_53

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(
            slice_129, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(
            slice_129, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_129

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_27 = paddle._C_ops.multiply(slice_130, slice_131)
        del slice_130, slice_131

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(
            slice_132, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(
            slice_132, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_132

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_28 = paddle._C_ops.multiply(slice_133, slice_134)
        del slice_133, slice_134

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(
            slice_135, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(
            slice_135, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_135

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_29 = paddle._C_ops.multiply(slice_136, slice_137)
        del slice_136, slice_137

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_40 = [multiply_27, multiply_28, multiply_29]
        del multiply_27, multiply_28, multiply_29

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_40, 0)
        del combine_40

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_31 = paddle._C_ops.split(reshape_68, stack_29, full_0)
        del reshape_68, stack_29

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_32,
            split_33,
            split_34,
        ) = split_31
        del split_31

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(add_170, full_10, float("0"), True)
        del add_170

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(scale_50, full_6, float("-1"), True)
        del scale_50

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(
            slice_138, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(
            slice_138, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_138

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_39 = paddle._C_ops.flatten(split_32, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_55 = paddle._C_ops.transpose(flatten_39, [0, 2, 1])
        del flatten_39

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_41 = [full_44, full_45, slice_139, slice_140]
        del slice_139, slice_140

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_30 = paddle._C_ops.stack(combine_41, 0)
        del combine_41

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_72 = paddle._C_ops.reshape(transpose_55, stack_30)
        del stack_30

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(
            scale_51, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_56 = paddle._C_ops.transpose(slice_141, [0, 2, 1, 3, 4])
        del slice_141

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_40 = paddle._C_ops.flatten(transpose_56, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_12 = paddle._C_ops.grid_sample(
            reshape_72, flatten_40, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(
            slice_142, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(
            slice_142, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_142

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_41 = paddle._C_ops.flatten(split_33, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_57 = paddle._C_ops.transpose(flatten_41, [0, 2, 1])
        del flatten_41

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_42 = [full_44, full_45, slice_143, slice_144]
        del slice_143, slice_144

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_31 = paddle._C_ops.stack(combine_42, 0)
        del combine_42

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(transpose_57, stack_31)
        del stack_31

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(
            scale_51, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_58 = paddle._C_ops.transpose(slice_145, [0, 2, 1, 3, 4])
        del slice_145

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_42 = paddle._C_ops.flatten(transpose_58, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_13 = paddle._C_ops.grid_sample(
            reshape_73, flatten_42, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(
            assign_value__16, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__16

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(
            slice_146, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_148 = paddle._C_ops.slice(
            slice_146, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_146

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_43 = paddle._C_ops.flatten(split_34, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_59 = paddle._C_ops.transpose(flatten_43, [0, 2, 1])
        del flatten_43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_43 = [full_44, full_45, slice_147, slice_148]
        del slice_147, slice_148

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_32 = paddle._C_ops.stack(combine_43, 0)
        del combine_43

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_74 = paddle._C_ops.reshape(transpose_59, stack_32)
        del stack_32

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_149 = paddle._C_ops.slice(
            scale_51, [3], full_int_array_3, full_int_array_425, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_60 = paddle._C_ops.transpose(slice_149, [0, 2, 1, 3, 4])
        del slice_149

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_44 = paddle._C_ops.flatten(transpose_60, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_14 = paddle._C_ops.grid_sample(
            reshape_74, flatten_44, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_71, [0, 2, 1, 3, 4])
        del reshape_71

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_61, full_int_array_426)

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_44 = [grid_sample_12, grid_sample_13, grid_sample_14]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_33 = paddle._C_ops.stack(combine_44, -2)
        del combine_44

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_45 = paddle._C_ops.flatten(stack_33, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_30 = paddle._C_ops.multiply(flatten_45, reshape_75)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_30, full_int_array_0, None, False)

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(sum_4, full_int_array_427)

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_76, [0, 2, 1])
        del reshape_76

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_117 = paddle._C_ops.matmul(transpose_62, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_171 = paddle._C_ops.add(matmul_117, parameter_28)
        del parameter_28

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_172 = paddle._C_ops.add(layer_norm_57, add_171)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_172, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_118 = paddle._C_ops.matmul(layer_norm_60, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_173 = paddle._C_ops.add(matmul_118, parameter_24)
        del parameter_24

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_37 = paddle._C_ops.relu(add_173)
        del add_173

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_119 = paddle._C_ops.matmul(relu_37, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_174 = paddle._C_ops.add(matmul_119, parameter_22)
        del parameter_22

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_175 = paddle._C_ops.add(layer_norm_60, add_174)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_175, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_120 = paddle._C_ops.matmul(layer_norm_63, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_176 = paddle._C_ops.add(matmul_120, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_38 = paddle._C_ops.relu(add_176)
        del add_176

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_121 = paddle._C_ops.matmul(relu_38, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_177 = paddle._C_ops.add(matmul_121, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_39 = paddle._C_ops.relu(add_177)
        del add_177

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_122 = paddle._C_ops.matmul(relu_39, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_178 = paddle._C_ops.add(matmul_122, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_27 = paddle._C_ops.clip(share_data__5, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_28 = paddle._C_ops.clip(clip_27, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(clip_27, full_13, float("1"), True)
        del clip_27

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_29 = paddle._C_ops.clip(scale_52, full_11, full_12)
        del scale_52

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_14 = paddle._C_ops.divide(clip_28, clip_29)
        del clip_28, clip_29

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_10 = paddle._C_ops.log(divide_14)
        del divide_14

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_179 = paddle._C_ops.add(add_178, log_10)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_10 = paddle._C_ops.sigmoid(add_179)
        del add_179

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_63, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_123 = paddle._C_ops.matmul(layer_norm_66, parameter_139, False, False)

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_180 = paddle._C_ops.add(matmul_123, parameter_138)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_124 = paddle._C_ops.matmul(layer_norm_66, parameter_129, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_181 = paddle._C_ops.add(matmul_124, parameter_128)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_40 = paddle._C_ops.relu(add_181)
        del add_181

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_125 = paddle._C_ops.matmul(relu_40, parameter_127, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_182 = paddle._C_ops.add(matmul_125, parameter_126)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_41 = paddle._C_ops.relu(add_182)
        del add_182

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_126 = paddle._C_ops.matmul(relu_41, parameter_125, False, False)

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_183 = paddle._C_ops.add(matmul_126, parameter_124)

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_6 = paddle._C_ops.bmm(add_183, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(bmm_6, full_int_array_415)

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_127 = paddle._C_ops.matmul(layer_norm_63, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_184 = paddle._C_ops.add(matmul_127, parameter_136)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_42 = paddle._C_ops.relu(add_184)
        del add_184

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_128 = paddle._C_ops.matmul(relu_42, parameter_135, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_185 = paddle._C_ops.add(matmul_128, parameter_134)

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_43 = paddle._C_ops.relu(add_185)
        del add_185

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_129 = paddle._C_ops.matmul(relu_43, parameter_133, False, False)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_186 = paddle._C_ops.add(matmul_129, parameter_132)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_30 = paddle._C_ops.clip(sigmoid_8, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_31 = paddle._C_ops.clip(clip_30, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(clip_30, full_13, float("1"), True)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_32 = paddle._C_ops.clip(scale_53, full_11, full_12)

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_15 = paddle._C_ops.divide(clip_31, clip_32)

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_11 = paddle._C_ops.log(divide_15)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_187 = paddle._C_ops.add(add_186, log_11)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_11 = paddle._C_ops.sigmoid(add_187)
        del add_187

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__6 = sigmoid_10.detach()

        # pd_op.unsqueeze: (2x400x1x4xf32) <- (2x400x4xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(share_data__6, full_int_array_3)

        # pd_op.matmul: (2x400x512xf32) <- (2x400x4xf32, 4x512xf32)
        matmul_130 = paddle._C_ops.matmul(share_data__6, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (2x400x512xf32) <- (2x400x512xf32, 512xf32)
        add_188 = paddle._C_ops.add(matmul_130, parameter_122)
        del parameter_122

        # pd_op.relu: (2x400x512xf32) <- (2x400x512xf32)
        relu_44 = paddle._C_ops.relu(add_188)
        del add_188

        # pd_op.matmul: (2x400x256xf32) <- (2x400x512xf32, 512x256xf32)
        matmul_131 = paddle._C_ops.matmul(relu_44, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_189 = paddle._C_ops.add(matmul_131, parameter_120)
        del parameter_120

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_190 = paddle._C_ops.add(layer_norm_63, add_189)

        # pd_op.where: (400x400xf32) <- (400x400xb, 400x400xf32, 400x400xf32)
        where_9 = paddle._C_ops.where(bitwise_not_0, full_38, full_39)
        del bitwise_not_0, full_38, full_39

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(
            data_11, [1], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_151 = paddle._C_ops.slice(
            data_12, [0], full_int_array_1, full_int_array_416, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_132 = paddle._C_ops.matmul(add_190, slice_150, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_191 = paddle._C_ops.add(matmul_132, slice_151)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_78 = paddle._C_ops.reshape(add_191, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_78, [0, 2, 1, 3])
        del reshape_78

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(
            data_11, [1], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(
            data_12, [0], full_int_array_416, full_int_array_418, [1], []
        )

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_133 = paddle._C_ops.matmul(add_190, slice_152, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_192 = paddle._C_ops.add(matmul_133, slice_153)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(add_192, full_int_array_417)

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_79, [0, 2, 1, 3])
        del reshape_79

        # pd_op.slice: (256x256xf32) <- (256x768xf32, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(
            data_11, [1], full_int_array_418, full_int_array_9, [1], []
        )
        del data_11

        # pd_op.slice: (256xf32) <- (768xf32, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(
            data_12, [0], full_int_array_418, full_int_array_9, [1], []
        )
        del data_12

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_134 = paddle._C_ops.matmul(layer_norm_63, slice_154, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_193 = paddle._C_ops.add(matmul_134, slice_155)

        # pd_op.reshape: (2x400x8x32xf32) <- (2x400x256xf32, 4xi64)
        reshape_80 = paddle._C_ops.reshape(add_193, full_int_array_417)
        del full_int_array_417

        # pd_op.transpose: (2x8x400x32xf32) <- (2x400x8x32xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_80, [0, 2, 1, 3])
        del reshape_80

        # pd_op.matmul: (2x8x400x400xf32) <- (2x8x400x32xf32, 2x8x400x32xf32)
        matmul_135 = paddle._C_ops.matmul(transpose_63, transpose_64, False, True)

        # pd_op.scale: (2x8x400x400xf32) <- (2x8x400x400xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(matmul_135, full_40, float("0"), True)
        del matmul_135

        # pd_op.add: (2x8x400x400xf32) <- (2x8x400x400xf32, 400x400xf32)
        add_194 = paddle._C_ops.add(scale_54, where_9)

        # pd_op.softmax: (2x8x400x400xf32) <- (2x8x400x400xf32)
        softmax_10 = paddle._C_ops.softmax(add_194, -1)
        del add_194

        # pd_op.matmul: (2x8x400x32xf32) <- (2x8x400x400xf32, 2x8x400x32xf32)
        matmul_136 = paddle._C_ops.matmul(softmax_10, transpose_65, False, False)

        # pd_op.transpose: (2x400x8x32xf32) <- (2x8x400x32xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_136, [0, 2, 1, 3])
        del matmul_136

        # pd_op.reshape: (2x400x256xf32) <- (2x400x8x32xf32, 3xi64)
        reshape_81 = paddle._C_ops.reshape(transpose_66, full_int_array_419)
        del full_int_array_419

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_137 = paddle._C_ops.matmul(reshape_81, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_195 = paddle._C_ops.add(matmul_137, parameter_18)
        del parameter_18

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_196 = paddle._C_ops.add(layer_norm_63, add_195)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_196, parameter_17, parameter_16, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_197 = paddle._C_ops.add(layer_norm_69, add_189)

        # pd_op.matmul: (2x12096x256xf32) <- (2x12096x256xf32, 256x256xf32)
        matmul_138 = paddle._C_ops.matmul(concat_0, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (2x12096x256xf32) <- (2x12096x256xf32, 256xf32)
        add_198 = paddle._C_ops.add(matmul_138, parameter_14)
        del parameter_14

        # pd_op.reshape: (2x12096x8x32xf32) <- (2x12096x256xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(add_198, full_int_array_420)
        del full_int_array_420

        # pd_op.matmul: (2x400x192xf32) <- (2x400x256xf32, 256x192xf32)
        matmul_139 = paddle._C_ops.matmul(add_197, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (2x400x192xf32) <- (2x400x192xf32, 192xf32)
        add_199 = paddle._C_ops.add(matmul_139, parameter_12)
        del parameter_12

        # pd_op.reshape: (2x400x8x3x4x2xf32) <- (2x400x192xf32, 6xi64)
        reshape_83 = paddle._C_ops.reshape(add_199, full_int_array_421)
        del full_int_array_421

        # pd_op.matmul: (2x400x96xf32) <- (2x400x256xf32, 256x96xf32)
        matmul_140 = paddle._C_ops.matmul(add_197, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (2x400x96xf32) <- (2x400x96xf32, 96xf32)
        add_200 = paddle._C_ops.add(matmul_140, parameter_10)
        del parameter_10

        # pd_op.reshape: (2x400x8x12xf32) <- (2x400x96xf32, 4xi64)
        reshape_84 = paddle._C_ops.reshape(add_200, full_int_array_422)
        del full_int_array_422

        # pd_op.softmax: (2x400x8x12xf32) <- (2x400x8x12xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_84, -1)
        del reshape_84

        # pd_op.reshape: (2x400x8x3x4xf32) <- (2x400x8x12xf32, 5xi64)
        reshape_85 = paddle._C_ops.reshape(softmax_11, full_int_array_423)
        del full_int_array_423

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_156 = paddle._C_ops.slice(
            unsqueeze_20, [3], full_int_array_1, full_int_array_3, [1], []
        )

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(slice_156, full_int_array_424)
        del slice_156

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(reshape_83, full_41, float("0"), True)
        del reshape_83

        # pd_op.slice: (2x400x1x2xf32) <- (2x400x1x4xf32, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(
            unsqueeze_20, [3], full_int_array_3, full_int_array_9, [1], []
        )
        del full_int_array_9, unsqueeze_20

        # pd_op.unsqueeze: (2x400x1x1x1x2xf32) <- (2x400x1x2xf32, 2xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(slice_157, full_int_array_424)
        del full_int_array_424, slice_157

        # pd_op.multiply: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 2x400x1x1x1x2xf32)
        multiply_31 = paddle._C_ops.multiply(scale_55, unsqueeze_22)

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(multiply_31, full_8, float("0"), True)
        del full_8, multiply_31

        # pd_op.add: (2x400x8x3x4x2xf32) <- (2x400x1x1x1x2xf32, 2x400x8x3x4x2xf32)
        add_201 = paddle._C_ops.add(unsqueeze_21, scale_56)

        # pd_op.full: (3x2xi64) <- ()
        full_54 = paddle._C_ops.full(
            [3, 2], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3x2xi64) <- (3x2xi64)
        assign_value__18 = paddle._C_ops.assign_value_(
            full_54,
            [3, 2],
            paddle.int64,
            [
                float("96"),
                float("96"),
                float("48"),
                float("48"),
                float("24"),
                float("24"),
            ],
            paddle.framework._current_expected_place(),
        )
        del full_54

        # pd_op.full: (3xi64) <- ()
        full_55 = paddle._C_ops.full(
            [3], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (3xi64) <- (3xi64)
        assign_value__19 = paddle._C_ops.assign_value_(
            full_55,
            [3],
            paddle.int64,
            [float("0"), float("9216"), float("11520")],
            paddle.framework._current_expected_place(),
        )
        del full_55

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_158 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_159 = paddle._C_ops.slice(
            slice_158, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(
            slice_158, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_158

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_32 = paddle._C_ops.multiply(slice_159, slice_160)
        del slice_159, slice_160

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(
            slice_161, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(
            slice_161, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_161

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_33 = paddle._C_ops.multiply(slice_162, slice_163)
        del slice_162, slice_163

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_164 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_3, full_int_array_425, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_165 = paddle._C_ops.slice(
            slice_164, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_166 = paddle._C_ops.slice(
            slice_164, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_164

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_34 = paddle._C_ops.multiply(slice_165, slice_166)
        del slice_165, slice_166

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_45 = [multiply_32, multiply_33, multiply_34]
        del multiply_32, multiply_33, multiply_34

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_34 = paddle._C_ops.stack(combine_45, 0)
        del combine_45

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (2x12096x8x32xf32, 3xi64, 1xi32)
        split_35 = paddle._C_ops.split(reshape_82, stack_34, full_0)
        del reshape_82, stack_34

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_36,
            split_37,
            split_38,
        ) = split_35
        del split_35

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(add_201, full_10, float("0"), True)
        del add_201, full_10

        # pd_op.scale: (2x400x8x3x4x2xf32) <- (2x400x8x3x4x2xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(scale_57, full_6, float("-1"), True)
        del scale_57

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(
            slice_167, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(
            slice_167, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_167

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_46 = paddle._C_ops.flatten(split_36, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_67 = paddle._C_ops.transpose(flatten_46, [0, 2, 1])
        del flatten_46

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_46 = [full_44, full_45, slice_168, slice_169]
        del slice_168, slice_169

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_35 = paddle._C_ops.stack(combine_46, 0)
        del combine_46

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_86 = paddle._C_ops.reshape(transpose_67, stack_35)
        del stack_35

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(
            scale_58, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_68 = paddle._C_ops.transpose(slice_170, [0, 2, 1, 3, 4])
        del slice_170

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_47 = paddle._C_ops.flatten(transpose_68, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_15 = paddle._C_ops.grid_sample(
            reshape_86, flatten_47, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_172 = paddle._C_ops.slice(
            slice_171, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_173 = paddle._C_ops.slice(
            slice_171, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del slice_171

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_48 = paddle._C_ops.flatten(split_37, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_69 = paddle._C_ops.transpose(flatten_48, [0, 2, 1])
        del flatten_48

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_47 = [full_44, full_45, slice_172, slice_173]
        del slice_172, slice_173

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_36 = paddle._C_ops.stack(combine_47, 0)
        del combine_47

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_87 = paddle._C_ops.reshape(transpose_69, stack_36)
        del stack_36

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(
            scale_58, [3], full_int_array_2, full_int_array_3, [1], [3]
        )

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_70 = paddle._C_ops.transpose(slice_174, [0, 2, 1, 3, 4])
        del slice_174

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_49 = paddle._C_ops.flatten(transpose_70, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_16 = paddle._C_ops.grid_sample(
            reshape_87, flatten_49, "bilinear", "zeros", False
        )

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(
            assign_value__18, [0], full_int_array_3, full_int_array_425, [1], [0]
        )
        del assign_value__18

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(
            slice_175, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(
            slice_175, [0], full_int_array_2, full_int_array_3, [1], [0]
        )
        del full_int_array_2, slice_175

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_50 = paddle._C_ops.flatten(split_38, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_71 = paddle._C_ops.transpose(flatten_50, [0, 2, 1])
        del flatten_50

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_48 = [full_44, full_45, slice_176, slice_177]
        del full_44, full_45, slice_176, slice_177

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_37 = paddle._C_ops.stack(combine_48, 0)
        del combine_48

        # pd_op.reshape: (16x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(transpose_71, stack_37)
        del stack_37

        # pd_op.slice: (2x400x8x4x2xf32) <- (2x400x8x3x4x2xf32, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(
            scale_58, [3], full_int_array_3, full_int_array_425, [1], [3]
        )
        del full_int_array_3, full_int_array_425

        # pd_op.transpose: (2x8x400x4x2xf32) <- (2x400x8x4x2xf32)
        transpose_72 = paddle._C_ops.transpose(slice_178, [0, 2, 1, 3, 4])
        del slice_178

        # pd_op.flatten: (16x400x4x2xf32) <- (2x8x400x4x2xf32)
        flatten_51 = paddle._C_ops.flatten(transpose_72, 0, 1)

        # pd_op.grid_sample: (16x32x400x4xf32) <- (16x32x-1x-1xf32, 16x400x4x2xf32)
        grid_sample_17 = paddle._C_ops.grid_sample(
            reshape_88, flatten_51, "bilinear", "zeros", False
        )

        # pd_op.transpose: (2x8x400x3x4xf32) <- (2x400x8x3x4xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_85, [0, 2, 1, 3, 4])
        del reshape_85

        # pd_op.reshape: (16x1x400x12xf32) <- (2x8x400x3x4xf32, 4xi64)
        reshape_89 = paddle._C_ops.reshape(transpose_73, full_int_array_426)
        del full_int_array_426

        # builtin.combine: ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32]) <- (16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32)
        combine_49 = [grid_sample_15, grid_sample_16, grid_sample_17]

        # pd_op.stack: (16x32x400x3x4xf32) <- ([16x32x400x4xf32, 16x32x400x4xf32, 16x32x400x4xf32])
        stack_38 = paddle._C_ops.stack(combine_49, -2)
        del combine_49

        # pd_op.flatten: (16x32x400x12xf32) <- (16x32x400x3x4xf32)
        flatten_52 = paddle._C_ops.flatten(stack_38, 3, 4)

        # pd_op.multiply: (16x32x400x12xf32) <- (16x32x400x12xf32, 16x1x400x12xf32)
        multiply_35 = paddle._C_ops.multiply(flatten_52, reshape_89)

        # pd_op.sum: (16x32x400xf32) <- (16x32x400x12xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_35, full_int_array_0, None, False)
        del full_int_array_0

        # pd_op.reshape: (2x256x400xf32) <- (16x32x400xf32, 3xi64)
        reshape_90 = paddle._C_ops.reshape(sum_5, full_int_array_427)
        del full_int_array_427

        # pd_op.transpose: (2x400x256xf32) <- (2x256x400xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_90, [0, 2, 1])
        del reshape_90

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_141 = paddle._C_ops.matmul(transpose_74, parameter_9, False, False)
        del parameter_9

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_202 = paddle._C_ops.add(matmul_141, parameter_8)
        del parameter_8

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_203 = paddle._C_ops.add(layer_norm_69, add_202)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_203, parameter_7, parameter_6, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_6, parameter_7

        # pd_op.matmul: (2x400x1024xf32) <- (2x400x256xf32, 256x1024xf32)
        matmul_142 = paddle._C_ops.matmul(layer_norm_72, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (2x400x1024xf32) <- (2x400x1024xf32, 1024xf32)
        add_204 = paddle._C_ops.add(matmul_142, parameter_4)
        del parameter_4

        # pd_op.relu: (2x400x1024xf32) <- (2x400x1024xf32)
        relu_45 = paddle._C_ops.relu(add_204)
        del add_204

        # pd_op.matmul: (2x400x256xf32) <- (2x400x1024xf32, 1024x256xf32)
        matmul_143 = paddle._C_ops.matmul(relu_45, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_205 = paddle._C_ops.add(matmul_143, parameter_2)
        del parameter_2

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 2x400x256xf32)
        add_206 = paddle._C_ops.add(layer_norm_72, add_205)

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_206, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_0, parameter_1

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_144 = paddle._C_ops.matmul(layer_norm_75, parameter_137, False, False)

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_207 = paddle._C_ops.add(matmul_144, parameter_136)
        del matmul_144

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_46 = paddle._C_ops.relu(add_207)
        del add_207

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_145 = paddle._C_ops.matmul(relu_46, parameter_135, False, False)
        del relu_46

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_208 = paddle._C_ops.add(matmul_145, parameter_134)
        del matmul_145

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_47 = paddle._C_ops.relu(add_208)
        del add_208

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_146 = paddle._C_ops.matmul(relu_47, parameter_133, False, False)
        del relu_47

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_209 = paddle._C_ops.add(matmul_146, parameter_132)
        del matmul_146

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_33 = paddle._C_ops.clip(share_data__6, full_5, full_6)

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_34 = paddle._C_ops.clip(clip_33, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(clip_33, full_13, float("1"), True)
        del clip_33

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_35 = paddle._C_ops.clip(scale_59, full_11, full_12)
        del scale_59

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_16 = paddle._C_ops.divide(clip_34, clip_35)
        del clip_34, clip_35

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_12 = paddle._C_ops.log(divide_16)
        del divide_16

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_210 = paddle._C_ops.add(add_209, log_12)
        del add_209, log_12

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_12 = paddle._C_ops.sigmoid(add_210)
        del add_210

        # pd_op.layer_norm: (2x400x256xf32, 2x400xf32, 2x400xf32) <- (2x400x256xf32, 256xf32, 256xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_75, parameter_131, parameter_130, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_130, parameter_131

        # pd_op.matmul: (2x400x2xf32) <- (2x400x256xf32, 256x2xf32)
        matmul_147 = paddle._C_ops.matmul(layer_norm_78, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (2x400x2xf32) <- (2x400x2xf32, 2xf32)
        add_211 = paddle._C_ops.add(matmul_147, parameter_138)
        del parameter_138

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_148 = paddle._C_ops.matmul(layer_norm_78, parameter_129, False, False)
        del parameter_129

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_212 = paddle._C_ops.add(matmul_148, parameter_128)
        del parameter_128

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_48 = paddle._C_ops.relu(add_212)
        del add_212

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_149 = paddle._C_ops.matmul(relu_48, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_213 = paddle._C_ops.add(matmul_149, parameter_126)
        del parameter_126

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_49 = paddle._C_ops.relu(add_213)
        del add_213

        # pd_op.matmul: (2x400x32xf32) <- (2x400x256xf32, 256x32xf32)
        matmul_150 = paddle._C_ops.matmul(relu_49, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (2x400x32xf32) <- (2x400x32xf32, 32xf32)
        add_214 = paddle._C_ops.add(matmul_150, parameter_124)
        del parameter_124

        # pd_op.bmm: (2x400x36864xf32) <- (2x400x32xf32, 2x32x36864xf32)
        bmm_7 = paddle._C_ops.bmm(add_214, flatten_6)

        # pd_op.reshape: (2x400x192x192xf32) <- (2x400x36864xf32, 4xi64)
        reshape_91 = paddle._C_ops.reshape(bmm_7, full_int_array_415)
        del full_int_array_415

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_151 = paddle._C_ops.matmul(layer_norm_75, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_215 = paddle._C_ops.add(matmul_151, parameter_136)
        del parameter_136

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_50 = paddle._C_ops.relu(add_215)
        del add_215

        # pd_op.matmul: (2x400x256xf32) <- (2x400x256xf32, 256x256xf32)
        matmul_152 = paddle._C_ops.matmul(relu_50, parameter_135, False, False)
        del parameter_135

        # pd_op.add: (2x400x256xf32) <- (2x400x256xf32, 256xf32)
        add_216 = paddle._C_ops.add(matmul_152, parameter_134)
        del parameter_134

        # pd_op.relu: (2x400x256xf32) <- (2x400x256xf32)
        relu_51 = paddle._C_ops.relu(add_216)
        del add_216

        # pd_op.matmul: (2x400x4xf32) <- (2x400x256xf32, 256x4xf32)
        matmul_153 = paddle._C_ops.matmul(relu_51, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 4xf32)
        add_217 = paddle._C_ops.add(matmul_153, parameter_132)
        del parameter_132

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_36 = paddle._C_ops.clip(sigmoid_10, full_5, full_6)
        del full_5, full_6

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_37 = paddle._C_ops.clip(clip_36, full_11, full_12)

        # pd_op.scale: (2x400x4xf32) <- (2x400x4xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(clip_36, full_13, float("1"), True)
        del full_13

        # pd_op.clip: (2x400x4xf32) <- (2x400x4xf32, 1xf32, 1xf32)
        clip_38 = paddle._C_ops.clip(scale_60, full_11, full_12)
        del full_11, full_12

        # pd_op.divide: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        divide_17 = paddle._C_ops.divide(clip_37, clip_38)

        # pd_op.log: (2x400x4xf32) <- (2x400x4xf32)
        log_13 = paddle._C_ops.log(divide_17)

        # pd_op.add: (2x400x4xf32) <- (2x400x4xf32, 2x400x4xf32)
        add_218 = paddle._C_ops.add(add_217, log_13)

        # pd_op.sigmoid: (2x400x4xf32) <- (2x400x4xf32)
        sigmoid_13 = paddle._C_ops.sigmoid(add_218)
        del add_218

        # pd_op.share_data_: (2x400x4xf32) <- (2x400x4xf32)
        share_data__7 = sigmoid_12.detach()
        del sigmoid_12

        # builtin.combine: ([2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32]) <- (2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32)
        combine_50 = [
            sigmoid_3,
            sigmoid_5,
            sigmoid_7,
            sigmoid_9,
            sigmoid_11,
            sigmoid_13,
        ]

        # pd_op.stack: (6x2x400x4xf32) <- ([2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32, 2x400x4xf32])
        stack_1 = paddle._C_ops.stack(combine_50, 0)
        del combine_50

        # builtin.combine: ([2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32]) <- (2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32)
        combine_51 = [add_60, add_87, add_118, add_149, add_180, add_211]

        # pd_op.stack: (6x2x400x2xf32) <- ([2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32, 2x400x2xf32])
        stack_0 = paddle._C_ops.stack(combine_51, 0)
        del combine_51

        # builtin.combine: ([2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32]) <- (2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32)
        combine_52 = [
            reshape_21,
            reshape_35,
            reshape_49,
            reshape_63,
            reshape_77,
            reshape_91,
        ]

        # pd_op.stack: (6x2x400x192x192xf32) <- ([2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32, 2x400x192x192xf32])
        stack_2 = paddle._C_ops.stack(combine_52, 0)
        del (
            add_10,
            add_100,
            add_102,
            add_103,
            add_104,
            add_105,
            add_106,
            add_107,
            add_109,
            add_11,
            add_110,
            add_112,
            add_113,
            add_116,
            add_118,
            add_121,
            add_124,
            add_127,
            add_128,
            add_129,
            add_13,
            add_130,
            add_131,
            add_133,
            add_134,
            add_135,
            add_136,
            add_137,
            add_138,
            add_140,
            add_141,
            add_143,
            add_144,
            add_147,
            add_149,
            add_152,
            add_155,
            add_158,
            add_159,
            add_160,
            add_161,
            add_162,
            add_164,
            add_165,
            add_166,
            add_167,
            add_168,
            add_169,
            add_17,
            add_171,
            add_172,
            add_174,
            add_175,
            add_178,
            add_18,
            add_180,
            add_183,
            add_186,
            add_189,
            add_190,
            add_191,
            add_192,
            add_193,
            add_195,
            add_196,
            add_197,
            add_198,
            add_199,
            add_200,
            add_202,
            add_203,
            add_205,
            add_206,
            add_21,
            add_211,
            add_214,
            add_217,
            add_36,
            add_38,
            add_39,
            add_40,
            add_41,
            add_42,
            add_44,
            add_45,
            add_46,
            add_47,
            add_48,
            add_49,
            add_51,
            add_52,
            add_54,
            add_55,
            add_58,
            add_60,
            add_63,
            add_65,
            add_66,
            add_67,
            add_68,
            add_69,
            add_71,
            add_72,
            add_73,
            add_74,
            add_75,
            add_76,
            add_78,
            add_79,
            add_81,
            add_82,
            add_85,
            add_87,
            add_9,
            add_90,
            add_93,
            add_96,
            add_97,
            add_98,
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
            assign_171,
            assign_172,
            assign_173,
            assign_174,
            assign_175,
            assign_176,
            assign_177,
            assign_178,
            assign_179,
            assign_18,
            assign_180,
            assign_181,
            assign_182,
            assign_183,
            assign_184,
            assign_185,
            assign_186,
            assign_187,
            assign_188,
            assign_189,
            assign_19,
            assign_190,
            assign_191,
            assign_192,
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
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            bmm_0,
            bmm_1,
            bmm_2,
            bmm_3,
            bmm_4,
            bmm_5,
            bmm_6,
            bmm_7,
            cast_10,
            clip_12,
            clip_13,
            clip_14,
            clip_18,
            clip_19,
            clip_20,
            clip_24,
            clip_25,
            clip_26,
            clip_30,
            clip_31,
            clip_32,
            clip_36,
            clip_37,
            clip_38,
            combine_52,
            concat_0,
            concat_2,
            concat_7,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            divide_11,
            divide_13,
            divide_15,
            divide_17,
            divide_9,
            flatten_12,
            flatten_14,
            flatten_16,
            flatten_17,
            flatten_19,
            flatten_21,
            flatten_23,
            flatten_24,
            flatten_26,
            flatten_28,
            flatten_30,
            flatten_31,
            flatten_33,
            flatten_35,
            flatten_37,
            flatten_38,
            flatten_40,
            flatten_42,
            flatten_44,
            flatten_45,
            flatten_47,
            flatten_49,
            flatten_5,
            flatten_51,
            flatten_52,
            flatten_6,
            full_0,
            full_14,
            full_40,
            full_41,
            full_int_array_416,
            full_int_array_418,
            gather_0,
            gather_nd_0,
            grid_sample_0,
            grid_sample_1,
            grid_sample_10,
            grid_sample_11,
            grid_sample_12,
            grid_sample_13,
            grid_sample_14,
            grid_sample_15,
            grid_sample_16,
            grid_sample_17,
            grid_sample_2,
            grid_sample_3,
            grid_sample_4,
            grid_sample_5,
            grid_sample_6,
            grid_sample_7,
            grid_sample_8,
            grid_sample_9,
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
            layer_norm_9,
            log_10,
            log_11,
            log_13,
            log_3,
            log_4,
            log_5,
            log_6,
            log_7,
            log_8,
            log_9,
            matmul_0,
            matmul_10,
            matmul_100,
            matmul_101,
            matmul_102,
            matmul_103,
            matmul_104,
            matmul_105,
            matmul_106,
            matmul_107,
            matmul_108,
            matmul_109,
            matmul_11,
            matmul_110,
            matmul_113,
            matmul_114,
            matmul_115,
            matmul_116,
            matmul_117,
            matmul_118,
            matmul_119,
            matmul_12,
            matmul_120,
            matmul_121,
            matmul_122,
            matmul_123,
            matmul_124,
            matmul_125,
            matmul_126,
            matmul_127,
            matmul_128,
            matmul_129,
            matmul_13,
            matmul_130,
            matmul_131,
            matmul_132,
            matmul_133,
            matmul_134,
            matmul_137,
            matmul_138,
            matmul_139,
            matmul_14,
            matmul_140,
            matmul_141,
            matmul_142,
            matmul_143,
            matmul_147,
            matmul_148,
            matmul_149,
            matmul_15,
            matmul_150,
            matmul_151,
            matmul_152,
            matmul_153,
            matmul_16,
            matmul_17,
            matmul_2,
            matmul_20,
            matmul_21,
            matmul_22,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_29,
            matmul_3,
            matmul_30,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_34,
            matmul_35,
            matmul_36,
            matmul_37,
            matmul_38,
            matmul_4,
            matmul_41,
            matmul_42,
            matmul_43,
            matmul_44,
            matmul_45,
            matmul_46,
            matmul_47,
            matmul_48,
            matmul_49,
            matmul_5,
            matmul_50,
            matmul_51,
            matmul_52,
            matmul_53,
            matmul_54,
            matmul_55,
            matmul_56,
            matmul_57,
            matmul_58,
            matmul_59,
            matmul_6,
            matmul_60,
            matmul_61,
            matmul_62,
            matmul_65,
            matmul_66,
            matmul_67,
            matmul_68,
            matmul_69,
            matmul_7,
            matmul_70,
            matmul_71,
            matmul_72,
            matmul_73,
            matmul_74,
            matmul_75,
            matmul_76,
            matmul_77,
            matmul_78,
            matmul_79,
            matmul_8,
            matmul_80,
            matmul_81,
            matmul_82,
            matmul_83,
            matmul_84,
            matmul_85,
            matmul_86,
            matmul_89,
            matmul_9,
            matmul_90,
            matmul_91,
            matmul_92,
            matmul_93,
            matmul_94,
            matmul_95,
            matmul_96,
            matmul_97,
            matmul_98,
            matmul_99,
            multiply_10,
            multiply_15,
            multiply_20,
            multiply_25,
            multiply_30,
            multiply_35,
            relu_0,
            relu_1,
            relu_10,
            relu_11,
            relu_12,
            relu_13,
            relu_14,
            relu_15,
            relu_16,
            relu_17,
            relu_18,
            relu_19,
            relu_2,
            relu_20,
            relu_21,
            relu_22,
            relu_23,
            relu_24,
            relu_25,
            relu_26,
            relu_27,
            relu_28,
            relu_29,
            relu_3,
            relu_30,
            relu_31,
            relu_32,
            relu_33,
            relu_34,
            relu_35,
            relu_36,
            relu_37,
            relu_38,
            relu_39,
            relu_4,
            relu_40,
            relu_41,
            relu_42,
            relu_43,
            relu_44,
            relu_45,
            relu_48,
            relu_49,
            relu_5,
            relu_50,
            relu_51,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
            reshape_11,
            reshape_16,
            reshape_17,
            reshape_18,
            reshape_19,
            reshape_21,
            reshape_25,
            reshape_3,
            reshape_30,
            reshape_31,
            reshape_32,
            reshape_33,
            reshape_35,
            reshape_39,
            reshape_44,
            reshape_45,
            reshape_46,
            reshape_47,
            reshape_49,
            reshape_53,
            reshape_58,
            reshape_59,
            reshape_60,
            reshape_61,
            reshape_63,
            reshape_67,
            reshape_7,
            reshape_72,
            reshape_73,
            reshape_74,
            reshape_75,
            reshape_77,
            reshape_81,
            reshape_86,
            reshape_87,
            reshape_88,
            reshape_89,
            reshape_91,
            scale_20,
            scale_21,
            scale_22,
            scale_24,
            scale_26,
            scale_27,
            scale_28,
            scale_30,
            scale_32,
            scale_33,
            scale_34,
            scale_35,
            scale_37,
            scale_39,
            scale_40,
            scale_41,
            scale_42,
            scale_44,
            scale_46,
            scale_47,
            scale_48,
            scale_49,
            scale_51,
            scale_53,
            scale_54,
            scale_55,
            scale_56,
            scale_58,
            scale_60,
            set_value__1,
            set_value__202,
            set_value_with_tensor__2,
            set_value_with_tensor__3,
            share_data__0,
            share_data__2,
            share_data__3,
            share_data__4,
            share_data__5,
            share_data__6,
            sigmoid_10,
            sigmoid_11,
            sigmoid_13,
            sigmoid_2,
            sigmoid_3,
            sigmoid_4,
            sigmoid_5,
            sigmoid_6,
            sigmoid_7,
            sigmoid_8,
            sigmoid_9,
            slice_10,
            slice_121,
            slice_122,
            slice_123,
            slice_124,
            slice_125,
            slice_126,
            slice_150,
            slice_151,
            slice_152,
            slice_153,
            slice_154,
            slice_155,
            slice_34,
            slice_35,
            slice_36,
            slice_37,
            slice_38,
            slice_39,
            slice_5,
            slice_6,
            slice_63,
            slice_64,
            slice_65,
            slice_66,
            slice_67,
            slice_68,
            slice_7,
            slice_8,
            slice_9,
            slice_92,
            slice_93,
            slice_94,
            slice_95,
            slice_96,
            slice_97,
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
            split_16,
            split_17,
            split_18,
            split_20,
            split_21,
            split_22,
            split_24,
            split_25,
            split_26,
            split_28,
            split_29,
            split_30,
            split_32,
            split_33,
            split_34,
            split_36,
            split_37,
            split_38,
            stack_13,
            stack_18,
            stack_23,
            stack_28,
            stack_33,
            stack_38,
            stack_7,
            sum_0,
            sum_1,
            sum_2,
            sum_3,
            sum_4,
            sum_5,
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
            transpose_69,
            transpose_7,
            transpose_70,
            transpose_71,
            transpose_72,
            transpose_73,
            transpose_74,
            transpose_8,
            transpose_9,
            unsqueeze_10,
            unsqueeze_12,
            unsqueeze_13,
            unsqueeze_15,
            unsqueeze_16,
            unsqueeze_18,
            unsqueeze_19,
            unsqueeze_21,
            unsqueeze_22,
            unsqueeze_6,
            unsqueeze_7,
            unsqueeze_9,
            where_0,
            where_1,
            where_4,
            where_5,
            where_6,
            where_7,
            where_8,
            where_9,
        )

        return (
            stack_0,
            stack_1,
            stack_2,
            add_0,
            sigmoid_0,
            add_1,
            sigmoid_1,
            reshape_0,
            split_0,
            split_1,
        )
