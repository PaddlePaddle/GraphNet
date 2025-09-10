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
        # pd_op.conv2d: (64x96x56x56xf32) <- (64x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_12, parameter_160, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_12, parameter_160

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_159, full_int_array_0)
        del full_int_array_0, parameter_159

        # pd_op.add: (64x96x56x56xf32) <- (64x96x56x56xf32, 1x96x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.flatten: (64x96x3136xf32) <- (64x96x56x56xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)

        # pd_op.transpose: (64x3136x96xf32) <- (64x96x3136xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (64x3136x96xf32, 64x3136xf32, 64x3136xf32) <- (64x3136x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_158, parameter_157, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_157, parameter_158

        # pd_op.layer_norm: (64x3136x96xf32, 64x3136xf32, 64x3136xf32) <- (64x3136x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_156, parameter_155, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_155, parameter_156

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [64, 56, 56, 96]

        # pd_op.reshape: (64x56x56x96xf32) <- (64x3136x96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(layer_norm_3, full_int_array_1)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_2 = [64, 8, 7, 8, 7, 96]

        # pd_op.reshape: (64x8x7x8x7x96xf32) <- (64x56x56x96xf32, 6xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, full_int_array_2)

        # pd_op.transpose: (64x8x8x7x7x96xf32) <- (64x8x7x8x7x96xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_2, [0, 1, 3, 2, 4, 5])
        del reshape_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [-1, 7, 7, 96]

        # pd_op.reshape: (4096x7x7x96xf32) <- (64x8x8x7x7x96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_1, full_int_array_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [-1, 49, 96]

        # pd_op.reshape: (4096x49x96xf32) <- (4096x7x7x96xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(reshape_3, full_int_array_4)

        # pd_op.matmul: (4096x49x288xf32) <- (4096x49x96xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_4, parameter_154, False, False)
        del parameter_154

        # pd_op.add: (4096x49x288xf32) <- (4096x49x288xf32, 288xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_153)
        del parameter_153

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_5 = [4096, 49, 3, 3, 32]

        # pd_op.reshape: (4096x49x3x3x32xf32) <- (4096x49x288xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_2, full_int_array_5)

        # pd_op.transpose: (3x4096x3x49x32xf32) <- (4096x49x3x3x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_11 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_12 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_13 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_14 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_15 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_16 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_17 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_6

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_7

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_7

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_55 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_56 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_57 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_58 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_59 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_60 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_61 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_62 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_63 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_64 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_65 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_66 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_67 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_68 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_69 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_70 = full_int_array_8

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_71 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_72 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_73 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_74 = full_int_array_9

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

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_82 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_83 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_84 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_85 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_86 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_87 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_88 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_89 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_90 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_91 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_92 = full_0

        # pd_op.scale: (4096x3x49x32xf32) <- (4096x3x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_0, float("0"), True)
        del slice_0

        # pd_op.transpose: (4096x3x32x49xf32) <- (4096x3x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_1, [0, 1, 3, 2])
        del slice_1

        # pd_op.matmul: (4096x3x49x49xf32) <- (4096x3x49x32xf32, 4096x3x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_3, False, False)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [-1]

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_6 = paddle._C_ops.reshape(data_14, full_int_array_10)
        del data_14

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, reshape_6, 0)
        del data_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_11 = [49, 49, -1]

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(index_select_0, full_int_array_11)

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_7, [2, 0, 1])
        del reshape_7

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_4, full_int_array_6)

        # pd_op.add: (4096x3x49x49xf32) <- (4096x3x49x49xf32, 1x3x49x49xf32)
        add_3 = paddle._C_ops.add(matmul_1, unsqueeze_0)

        # pd_op.softmax: (4096x3x49x49xf32) <- (4096x3x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.matmul: (4096x3x49x32xf32) <- (4096x3x49x49xf32, 4096x3x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_2, False, False)

        # pd_op.transpose: (4096x49x3x32xf32) <- (4096x3x49x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_12 = [4096, 49, 96]

        # pd_op.reshape: (4096x49x96xf32) <- (4096x49x3x32xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_5, full_int_array_12)

        # pd_op.matmul: (4096x49x96xf32) <- (4096x49x96xf32, 96x96xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_8, parameter_152, False, False)
        del parameter_152

        # pd_op.add: (4096x49x96xf32) <- (4096x49x96xf32, 96xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_151)
        del parameter_151

        # pd_op.reshape: (4096x7x7x96xf32) <- (4096x49x96xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_4, full_int_array_3)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_13 = [-1, 8, 8, 7, 7, 96]

        # pd_op.reshape: (64x8x8x7x7x96xf32) <- (4096x7x7x96xf32, 6xi64)
        reshape_10 = paddle._C_ops.reshape(reshape_9, full_int_array_13)

        # pd_op.transpose: (64x8x7x8x7x96xf32) <- (64x8x8x7x7x96xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_10, [0, 1, 3, 2, 4, 5])
        del reshape_10

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [-1, 56, 56, 96]

        # pd_op.reshape: (64x56x56x96xf32) <- (64x8x7x8x7x96xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_6, full_int_array_14)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_15 = [64, 3136, 96]

        # pd_op.reshape: (64x3136x96xf32) <- (64x56x56x96xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(reshape_11, full_int_array_15)

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 64x3136x96xf32)
        add_5 = paddle._C_ops.add(layer_norm_0, reshape_12)

        # pd_op.layer_norm: (64x3136x96xf32, 64x3136xf32, 64x3136xf32) <- (64x3136x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_150, parameter_149, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_149, parameter_150

        # pd_op.matmul: (64x3136x384xf32) <- (64x3136x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_148, False, False)
        del parameter_148

        # pd_op.add: (64x3136x384xf32) <- (64x3136x384xf32, 384xf32)
        add_6 = paddle._C_ops.add(matmul_4, parameter_147)
        del parameter_147

        # pd_op.gelu: (64x3136x384xf32) <- (64x3136x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)

        # pd_op.matmul: (64x3136x96xf32) <- (64x3136x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_146, False, False)
        del parameter_146

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 96xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_145)
        del parameter_145

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 64x3136x96xf32)
        add_8 = paddle._C_ops.add(add_5, add_7)

        # pd_op.layer_norm: (64x3136x96xf32, 64x3136xf32, 64x3136xf32) <- (64x3136x96xf32, 96xf32, 96xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_144, parameter_143, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_143, parameter_144

        # pd_op.reshape: (64x56x56x96xf32) <- (64x3136x96xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(layer_norm_9, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [-3, -3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_93 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_94 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_95 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_96 = full_int_array_16

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_97 = full_int_array_16

        # pd_op.roll: (64x56x56x96xf32) <- (64x56x56x96xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(reshape_13, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x8x7x8x7x96xf32) <- (64x56x56x96xf32, 6xi64)
        reshape_14 = paddle._C_ops.reshape(roll_0, full_int_array_2)
        del full_int_array_2

        # pd_op.transpose: (64x8x8x7x7x96xf32) <- (64x8x7x8x7x96xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_14, [0, 1, 3, 2, 4, 5])
        del reshape_14

        # pd_op.reshape: (4096x7x7x96xf32) <- (64x8x8x7x7x96xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_7, full_int_array_3)

        # pd_op.reshape: (4096x49x96xf32) <- (4096x7x7x96xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(reshape_15, full_int_array_4)
        del full_int_array_4

        # pd_op.full: (1x56x56x1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1, 56, 56, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [0, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_98 = full_int_array_17

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_99 = full_int_array_17

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_100 = full_int_array_17

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_101 = full_int_array_19

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_102 = full_int_array_19

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_103 = full_int_array_19

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_104 = full_int_array_19

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_1,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [-7, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [-7, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            set_value__1,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [-3, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__2

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__4 = paddle._C_ops.set_value_(
            set_value__3,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [-3, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_27 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [2147483647, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_29 = [2147483647, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_30 = [2147483647, 2147483647]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_105 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_106 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_107 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_108 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_109 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_110 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_111 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_112 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_113 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_114 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_115 = full_int_array_30

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_116 = full_int_array_30

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__7

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_31 = [1, 8, 7, 8, 7, 1]

        # pd_op.reshape: (1x8x7x8x7x1xf32) <- (1x56x56x1xf32, 6xi64)
        reshape_17 = paddle._C_ops.reshape(set_value__8, full_int_array_31)
        del full_int_array_31

        # pd_op.transpose: (1x8x8x7x7x1xf32) <- (1x8x7x8x7x1xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_17, [0, 1, 3, 2, 4, 5])
        del reshape_17

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [-1, 7, 7, 1]

        # pd_op.reshape: (64x7x7x1xf32) <- (1x8x8x7x7x1xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_8, full_int_array_32)
        del transpose_8

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_33 = [-1, 49]

        # pd_op.reshape: (64x49xf32) <- (64x7x7x1xf32, 2xi64)
        reshape_19 = paddle._C_ops.reshape(reshape_18, full_int_array_33)
        del reshape_18

        # pd_op.unsqueeze: (64x1x49xf32) <- (64x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_19, full_int_array_7)

        # pd_op.unsqueeze: (64x49x1xf32) <- (64x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(reshape_19, full_int_array_8)
        del reshape_19

        # pd_op.subtract: (64x49x49xf32) <- (64x1x49xf32, 64x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_1, unsqueeze_2)
        del unsqueeze_1, unsqueeze_2

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_2)

        # pd_op.full: (64x49x49xf32) <- ()
        full_3 = paddle._C_ops.full(
            [64, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        where_0 = paddle._C_ops.where(not_equal_0, full_3, subtract_0)
        del full_3, not_equal_0, subtract_0

        # pd_op.equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        equal_0 = paddle._C_ops.equal(where_0, full_2)

        # pd_op.full: (64x49x49xf32) <- ()
        full_4 = paddle._C_ops.full(
            [64, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        where_1 = paddle._C_ops.where(equal_0, full_4, where_0)
        del equal_0, full_4, where_0

        # pd_op.matmul: (4096x49x288xf32) <- (4096x49x96xf32, 96x288xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_16, parameter_142, False, False)
        del parameter_142

        # pd_op.add: (4096x49x288xf32) <- (4096x49x288xf32, 288xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_141)
        del parameter_141

        # pd_op.reshape: (4096x49x3x3x32xf32) <- (4096x49x288xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_9, full_int_array_5)
        del full_int_array_5

        # pd_op.transpose: (3x4096x3x49x32xf32) <- (4096x49x3x3x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (4096x3x49x32xf32) <- (3x4096x3x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (4096x3x49x32xf32) <- (4096x3x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_3, full_0, float("0"), True)
        del slice_3

        # pd_op.transpose: (4096x3x32x49xf32) <- (4096x3x49x32xf32)
        transpose_10 = paddle._C_ops.transpose(slice_4, [0, 1, 3, 2])
        del slice_4

        # pd_op.matmul: (4096x3x49x49xf32) <- (4096x3x49x32xf32, 4096x3x32x49xf32)
        matmul_7 = paddle._C_ops.matmul(scale_1, transpose_10, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_21 = paddle._C_ops.reshape(data_24, full_int_array_10)
        del data_24

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, reshape_21, 0)
        del data_1

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(index_select_1, full_int_array_11)

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_22, [2, 0, 1])
        del reshape_22

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_11, full_int_array_6)

        # pd_op.add: (4096x3x49x49xf32) <- (4096x3x49x49xf32, 1x3x49x49xf32)
        add_10 = paddle._C_ops.add(matmul_7, unsqueeze_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_34 = [64, 64, 3, 49, 49]

        # pd_op.reshape: (64x64x3x49x49xf32) <- (4096x3x49x49xf32, 5xi64)
        reshape_23 = paddle._C_ops.reshape(add_10, full_int_array_34)
        del full_int_array_34

        # pd_op.unsqueeze: (64x1x49x49xf32) <- (64x49x49xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(where_1, full_int_array_7)
        del where_1

        # pd_op.unsqueeze: (1x64x1x49x49xf32) <- (64x1x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(unsqueeze_4, full_int_array_6)
        del unsqueeze_4

        # pd_op.add: (64x64x3x49x49xf32) <- (64x64x3x49x49xf32, 1x64x1x49x49xf32)
        add_11 = paddle._C_ops.add(reshape_23, unsqueeze_5)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [4096, 3, 49, 49]

        # pd_op.reshape: (4096x3x49x49xf32) <- (64x64x3x49x49xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_11, full_int_array_35)
        del full_int_array_35

        # pd_op.softmax: (4096x3x49x49xf32) <- (4096x3x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_24, -1)
        del reshape_24

        # pd_op.matmul: (4096x3x49x32xf32) <- (4096x3x49x49xf32, 4096x3x49x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_5, False, False)

        # pd_op.transpose: (4096x49x3x32xf32) <- (4096x3x49x32xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # pd_op.reshape: (4096x49x96xf32) <- (4096x49x3x32xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_12, full_int_array_12)
        del full_int_array_12

        # pd_op.matmul: (4096x49x96xf32) <- (4096x49x96xf32, 96x96xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_25, parameter_140, False, False)
        del parameter_140

        # pd_op.add: (4096x49x96xf32) <- (4096x49x96xf32, 96xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_139)
        del parameter_139

        # pd_op.reshape: (4096x7x7x96xf32) <- (4096x49x96xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_12, full_int_array_3)
        del full_int_array_3

        # pd_op.reshape: (64x8x8x7x7x96xf32) <- (4096x7x7x96xf32, 6xi64)
        reshape_27 = paddle._C_ops.reshape(reshape_26, full_int_array_13)
        del full_int_array_13

        # pd_op.transpose: (64x8x7x8x7x96xf32) <- (64x8x8x7x7x96xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_27, [0, 1, 3, 2, 4, 5])
        del reshape_27

        # pd_op.reshape: (64x56x56x96xf32) <- (64x8x7x8x7x96xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(transpose_13, full_int_array_14)
        del full_int_array_14

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [3, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_117 = full_int_array_36

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_118 = full_int_array_36

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_119 = full_int_array_36

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_120 = full_int_array_36

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_121 = full_int_array_36

        # pd_op.roll: (64x56x56x96xf32) <- (64x56x56x96xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_28, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x3136x96xf32) <- (64x56x56x96xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(roll_1, full_int_array_15)
        del full_int_array_15

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.981818"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_122 = full_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_37 = [64, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_13 = paddle._C_ops.add(full_5, uniform_0)
        del uniform_0

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_13)
        del add_13

        # pd_op.divide: (64x3136x96xf32) <- (64x3136x96xf32, xf32)
        divide_0 = paddle._C_ops.divide(reshape_29, full_5)

        # pd_op.multiply: (64x3136x96xf32) <- (64x3136x96xf32, 64x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 64x3136x96xf32)
        add_14 = paddle._C_ops.add(add_8, multiply_0)

        # pd_op.layer_norm: (64x3136x96xf32, 64x3136xf32, 64x3136xf32) <- (64x3136x96xf32, 96xf32, 96xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_138, parameter_137, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_137, parameter_138

        # pd_op.matmul: (64x3136x384xf32) <- (64x3136x96xf32, 96x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_136, False, False)
        del parameter_136

        # pd_op.add: (64x3136x384xf32) <- (64x3136x384xf32, 384xf32)
        add_15 = paddle._C_ops.add(matmul_10, parameter_135)
        del parameter_135

        # pd_op.gelu: (64x3136x384xf32) <- (64x3136x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_15, False)

        # pd_op.matmul: (64x3136x96xf32) <- (64x3136x384xf32, 384x96xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_134, False, False)
        del parameter_134

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 96xf32)
        add_16 = paddle._C_ops.add(matmul_11, parameter_133)
        del parameter_133

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_17 = paddle._C_ops.add(full_5, uniform_1)
        del uniform_1

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_17)
        del add_17

        # pd_op.divide: (64x3136x96xf32) <- (64x3136x96xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_16, full_5)

        # pd_op.multiply: (64x3136x96xf32) <- (64x3136x96xf32, 64x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (64x3136x96xf32) <- (64x3136x96xf32, 64x3136x96xf32)
        add_18 = paddle._C_ops.add(add_14, multiply_1)

        # pd_op.reshape: (64x56x56x96xf32) <- (64x3136x96xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_18, full_int_array_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [2, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_123 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_124 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_125 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_126 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_127 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_128 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_129 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_130 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_131 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_132 = full_int_array_38

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_133 = full_int_array_38

        # pd_op.strided_slice: (64x28x28x96xf32) <- (64x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_17, full_int_array_30, full_int_array_38
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_39 = [1, 0]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_134 = full_int_array_39

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_135 = full_int_array_39

        # pd_op.strided_slice: (64x28x28x96xf32) <- (64x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_39, full_int_array_30, full_int_array_38
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_40 = [0, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_136 = full_int_array_40

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_137 = full_int_array_40

        # pd_op.strided_slice: (64x28x28x96xf32) <- (64x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_40, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x28x28x96xf32) <- (64x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_19, full_int_array_30, full_int_array_38
        )

        # pd_op.reshape: (64x56x56x96xf32) <- (64x56x56x96xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(reshape_30, full_int_array_1)
        del full_int_array_1

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_138 = full_8

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_139 = full_8

        # builtin.combine: ([64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32]) <- (64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32)
        combine_0 = [strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3]

        # pd_op.concat: (64x28x28x384xf32) <- ([64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32, 64x28x28x96xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_8)
        del combine_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_41 = [64, -1, 384]

        # pd_op.reshape: (64x784x384xf32) <- (64x28x28x384xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(concat_0, full_int_array_41)
        del full_int_array_41

        # pd_op.layer_norm: (64x784x384xf32, 64x784xf32, 64x784xf32) <- (64x784x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_32, parameter_132, parameter_131, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_131, parameter_132

        # pd_op.matmul: (64x784x192xf32) <- (64x784x384xf32, 384x192xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_130, False, False)
        del parameter_130

        # pd_op.layer_norm: (64x784x192xf32, 64x784xf32, 64x784xf32) <- (64x784x192xf32, 192xf32, 192xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_12, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [64, 28, 28, 192]

        # pd_op.reshape: (64x28x28x192xf32) <- (64x784x192xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(layer_norm_18, full_int_array_42)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_43 = [64, 4, 7, 4, 7, 192]

        # pd_op.reshape: (64x4x7x4x7x192xf32) <- (64x28x28x192xf32, 6xi64)
        reshape_34 = paddle._C_ops.reshape(reshape_33, full_int_array_43)

        # pd_op.transpose: (64x4x4x7x7x192xf32) <- (64x4x7x4x7x192xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_34, [0, 1, 3, 2, 4, 5])
        del reshape_34

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [-1, 7, 7, 192]

        # pd_op.reshape: (1024x7x7x192xf32) <- (64x4x4x7x7x192xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_14, full_int_array_44)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_45 = [-1, 49, 192]

        # pd_op.reshape: (1024x49x192xf32) <- (1024x7x7x192xf32, 3xi64)
        reshape_36 = paddle._C_ops.reshape(reshape_35, full_int_array_45)

        # pd_op.matmul: (1024x49x576xf32) <- (1024x49x192xf32, 192x576xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_36, parameter_127, False, False)
        del parameter_127

        # pd_op.add: (1024x49x576xf32) <- (1024x49x576xf32, 576xf32)
        add_19 = paddle._C_ops.add(matmul_13, parameter_126)
        del parameter_126

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_46 = [1024, 49, 3, 6, 32]

        # pd_op.reshape: (1024x49x3x6x32xf32) <- (1024x49x576xf32, 5xi64)
        reshape_37 = paddle._C_ops.reshape(add_19, full_int_array_46)

        # pd_op.transpose: (3x1024x6x49x32xf32) <- (1024x49x3x6x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_37, [2, 0, 3, 1, 4])
        del reshape_37

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (1024x6x49x32xf32) <- (1024x6x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_6, full_0, float("0"), True)
        del slice_6

        # pd_op.transpose: (1024x6x32x49xf32) <- (1024x6x49x32xf32)
        transpose_16 = paddle._C_ops.transpose(slice_7, [0, 1, 3, 2])
        del slice_7

        # pd_op.matmul: (1024x6x49x49xf32) <- (1024x6x49x32xf32, 1024x6x32x49xf32)
        matmul_14 = paddle._C_ops.matmul(scale_2, transpose_16, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_38 = paddle._C_ops.reshape(data_13, full_int_array_10)
        del data_13

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_2 = paddle._C_ops.index_select(data_4, reshape_38, 0)
        del data_4

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(index_select_2, full_int_array_11)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_39, [2, 0, 1])
        del reshape_39

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(transpose_17, full_int_array_6)

        # pd_op.add: (1024x6x49x49xf32) <- (1024x6x49x49xf32, 1x6x49x49xf32)
        add_20 = paddle._C_ops.add(matmul_14, unsqueeze_6)

        # pd_op.softmax: (1024x6x49x49xf32) <- (1024x6x49x49xf32)
        softmax_2 = paddle._C_ops.softmax(add_20, -1)
        del add_20

        # pd_op.matmul: (1024x6x49x32xf32) <- (1024x6x49x49xf32, 1024x6x49x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_8, False, False)

        # pd_op.transpose: (1024x49x6x32xf32) <- (1024x6x49x32xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_47 = [1024, 49, 192]

        # pd_op.reshape: (1024x49x192xf32) <- (1024x49x6x32xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(transpose_18, full_int_array_47)

        # pd_op.matmul: (1024x49x192xf32) <- (1024x49x192xf32, 192x192xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_40, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (1024x49x192xf32) <- (1024x49x192xf32, 192xf32)
        add_21 = paddle._C_ops.add(matmul_16, parameter_124)
        del parameter_124

        # pd_op.reshape: (1024x7x7x192xf32) <- (1024x49x192xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_21, full_int_array_44)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_48 = [-1, 4, 4, 7, 7, 192]

        # pd_op.reshape: (64x4x4x7x7x192xf32) <- (1024x7x7x192xf32, 6xi64)
        reshape_42 = paddle._C_ops.reshape(reshape_41, full_int_array_48)

        # pd_op.transpose: (64x4x7x4x7x192xf32) <- (64x4x4x7x7x192xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_42, [0, 1, 3, 2, 4, 5])
        del reshape_42

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [-1, 28, 28, 192]

        # pd_op.reshape: (64x28x28x192xf32) <- (64x4x7x4x7x192xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_19, full_int_array_49)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_50 = [64, 784, 192]

        # pd_op.reshape: (64x784x192xf32) <- (64x28x28x192xf32, 3xi64)
        reshape_44 = paddle._C_ops.reshape(reshape_43, full_int_array_50)

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.963636"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_140 = full_9

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_22 = paddle._C_ops.add(full_9, uniform_2)
        del uniform_2

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (64x784x192xf32) <- (64x784x192xf32, xf32)
        divide_2 = paddle._C_ops.divide(reshape_44, full_9)

        # pd_op.multiply: (64x784x192xf32) <- (64x784x192xf32, 64x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 64x784x192xf32)
        add_23 = paddle._C_ops.add(matmul_12, multiply_2)

        # pd_op.layer_norm: (64x784x192xf32, 64x784xf32, 64x784xf32) <- (64x784x192xf32, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (64x784x768xf32) <- (64x784x192xf32, 192x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_21, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (64x784x768xf32) <- (64x784x768xf32, 768xf32)
        add_24 = paddle._C_ops.add(matmul_17, parameter_120)
        del parameter_120

        # pd_op.gelu: (64x784x768xf32) <- (64x784x768xf32)
        gelu_2 = paddle._C_ops.gelu(add_24, False)

        # pd_op.matmul: (64x784x192xf32) <- (64x784x768xf32, 768x192xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_119, False, False)
        del parameter_119

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 192xf32)
        add_25 = paddle._C_ops.add(matmul_18, parameter_118)
        del parameter_118

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_26 = paddle._C_ops.add(full_9, uniform_3)
        del uniform_3

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (64x784x192xf32) <- (64x784x192xf32, xf32)
        divide_3 = paddle._C_ops.divide(add_25, full_9)

        # pd_op.multiply: (64x784x192xf32) <- (64x784x192xf32, 64x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 64x784x192xf32)
        add_27 = paddle._C_ops.add(add_23, multiply_3)

        # pd_op.layer_norm: (64x784x192xf32, 64x784xf32, 64x784xf32) <- (64x784x192xf32, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_27, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.reshape: (64x28x28x192xf32) <- (64x784x192xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(layer_norm_24, full_int_array_42)

        # pd_op.roll: (64x28x28x192xf32) <- (64x28x28x192xf32, 2xi64)
        roll_2 = paddle._C_ops.roll(reshape_45, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x4x7x4x7x192xf32) <- (64x28x28x192xf32, 6xi64)
        reshape_46 = paddle._C_ops.reshape(roll_2, full_int_array_43)
        del full_int_array_43

        # pd_op.transpose: (64x4x4x7x7x192xf32) <- (64x4x7x4x7x192xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_46, [0, 1, 3, 2, 4, 5])
        del reshape_46

        # pd_op.reshape: (1024x7x7x192xf32) <- (64x4x4x7x7x192xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_20, full_int_array_44)

        # pd_op.reshape: (1024x49x192xf32) <- (1024x7x7x192xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(reshape_47, full_int_array_45)
        del full_int_array_45

        # pd_op.full: (1x28x28x1xf32) <- ()
        full_10 = paddle._C_ops.full(
            [1, 28, 28, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__9 = paddle._C_ops.set_value_(
            full_10,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_10

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__10 = paddle._C_ops.set_value_(
            set_value__9,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__9

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__11 = paddle._C_ops.set_value_(
            set_value__10,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__10

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__12 = paddle._C_ops.set_value_(
            set_value__11,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__11

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__13 = paddle._C_ops.set_value_(
            set_value__12,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__12

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__14 = paddle._C_ops.set_value_(
            set_value__13,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__13

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__15 = paddle._C_ops.set_value_(
            set_value__14,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__14

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__16 = paddle._C_ops.set_value_(
            set_value__15,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__15

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__17 = paddle._C_ops.set_value_(
            set_value__16,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__16

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_51 = [1, 4, 7, 4, 7, 1]

        # pd_op.reshape: (1x4x7x4x7x1xf32) <- (1x28x28x1xf32, 6xi64)
        reshape_49 = paddle._C_ops.reshape(set_value__17, full_int_array_51)
        del full_int_array_51

        # pd_op.transpose: (1x4x4x7x7x1xf32) <- (1x4x7x4x7x1xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_49, [0, 1, 3, 2, 4, 5])
        del reshape_49

        # pd_op.reshape: (16x7x7x1xf32) <- (1x4x4x7x7x1xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_21, full_int_array_32)
        del transpose_21

        # pd_op.reshape: (16x49xf32) <- (16x7x7x1xf32, 2xi64)
        reshape_51 = paddle._C_ops.reshape(reshape_50, full_int_array_33)
        del reshape_50

        # pd_op.unsqueeze: (16x1x49xf32) <- (16x49xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(reshape_51, full_int_array_7)

        # pd_op.unsqueeze: (16x49x1xf32) <- (16x49xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(reshape_51, full_int_array_8)
        del reshape_51

        # pd_op.subtract: (16x49x49xf32) <- (16x1x49xf32, 16x49x1xf32)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_7, unsqueeze_8)
        del unsqueeze_7, unsqueeze_8

        # pd_op.not_equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        not_equal_1 = paddle._C_ops.not_equal(subtract_1, full_2)

        # pd_op.full: (16x49x49xf32) <- ()
        full_11 = paddle._C_ops.full(
            [16, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        where_2 = paddle._C_ops.where(not_equal_1, full_11, subtract_1)
        del full_11, not_equal_1, subtract_1

        # pd_op.equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        equal_1 = paddle._C_ops.equal(where_2, full_2)

        # pd_op.full: (16x49x49xf32) <- ()
        full_12 = paddle._C_ops.full(
            [16, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        where_3 = paddle._C_ops.where(equal_1, full_12, where_2)
        del equal_1, full_12, where_2

        # pd_op.matmul: (1024x49x576xf32) <- (1024x49x192xf32, 192x576xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_48, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (1024x49x576xf32) <- (1024x49x576xf32, 576xf32)
        add_28 = paddle._C_ops.add(matmul_19, parameter_114)
        del parameter_114

        # pd_op.reshape: (1024x49x3x6x32xf32) <- (1024x49x576xf32, 5xi64)
        reshape_52 = paddle._C_ops.reshape(add_28, full_int_array_46)
        del full_int_array_46

        # pd_op.transpose: (3x1024x6x49x32xf32) <- (1024x49x3x6x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_52, [2, 0, 3, 1, 4])
        del reshape_52

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (1024x6x49x32xf32) <- (3x1024x6x49x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (1024x6x49x32xf32) <- (1024x6x49x32xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_9, full_0, float("0"), True)
        del slice_9

        # pd_op.transpose: (1024x6x32x49xf32) <- (1024x6x49x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])
        del slice_10

        # pd_op.matmul: (1024x6x49x49xf32) <- (1024x6x49x32xf32, 1024x6x32x49xf32)
        matmul_20 = paddle._C_ops.matmul(scale_3, transpose_23, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_53 = paddle._C_ops.reshape(data_15, full_int_array_10)
        del data_15

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_3 = paddle._C_ops.index_select(data_5, reshape_53, 0)
        del data_5

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_54 = paddle._C_ops.reshape(index_select_3, full_int_array_11)

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_54, [2, 0, 1])
        del reshape_54

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(transpose_24, full_int_array_6)

        # pd_op.add: (1024x6x49x49xf32) <- (1024x6x49x49xf32, 1x6x49x49xf32)
        add_29 = paddle._C_ops.add(matmul_20, unsqueeze_9)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_52 = [64, 16, 6, 49, 49]

        # pd_op.reshape: (64x16x6x49x49xf32) <- (1024x6x49x49xf32, 5xi64)
        reshape_55 = paddle._C_ops.reshape(add_29, full_int_array_52)
        del full_int_array_52

        # pd_op.unsqueeze: (16x1x49x49xf32) <- (16x49x49xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(where_3, full_int_array_7)
        del where_3

        # pd_op.unsqueeze: (1x16x1x49x49xf32) <- (16x1x49x49xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(unsqueeze_10, full_int_array_6)
        del unsqueeze_10

        # pd_op.add: (64x16x6x49x49xf32) <- (64x16x6x49x49xf32, 1x16x1x49x49xf32)
        add_30 = paddle._C_ops.add(reshape_55, unsqueeze_11)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1024, 6, 49, 49]

        # pd_op.reshape: (1024x6x49x49xf32) <- (64x16x6x49x49xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_30, full_int_array_53)
        del full_int_array_53

        # pd_op.softmax: (1024x6x49x49xf32) <- (1024x6x49x49xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_56, -1)
        del reshape_56

        # pd_op.matmul: (1024x6x49x32xf32) <- (1024x6x49x49xf32, 1024x6x49x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_11, False, False)

        # pd_op.transpose: (1024x49x6x32xf32) <- (1024x6x49x32xf32)
        transpose_25 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # pd_op.reshape: (1024x49x192xf32) <- (1024x49x6x32xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_25, full_int_array_47)
        del full_int_array_47

        # pd_op.matmul: (1024x49x192xf32) <- (1024x49x192xf32, 192x192xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_57, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (1024x49x192xf32) <- (1024x49x192xf32, 192xf32)
        add_31 = paddle._C_ops.add(matmul_22, parameter_112)
        del parameter_112

        # pd_op.reshape: (1024x7x7x192xf32) <- (1024x49x192xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_31, full_int_array_44)
        del full_int_array_44

        # pd_op.reshape: (64x4x4x7x7x192xf32) <- (1024x7x7x192xf32, 6xi64)
        reshape_59 = paddle._C_ops.reshape(reshape_58, full_int_array_48)
        del full_int_array_48

        # pd_op.transpose: (64x4x7x4x7x192xf32) <- (64x4x4x7x7x192xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_59, [0, 1, 3, 2, 4, 5])
        del reshape_59

        # pd_op.reshape: (64x28x28x192xf32) <- (64x4x7x4x7x192xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(transpose_26, full_int_array_49)
        del full_int_array_49

        # pd_op.roll: (64x28x28x192xf32) <- (64x28x28x192xf32, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape_60, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x784x192xf32) <- (64x28x28x192xf32, 3xi64)
        reshape_61 = paddle._C_ops.reshape(roll_3, full_int_array_50)
        del full_int_array_50

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.945455"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_141 = full_13

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_32 = paddle._C_ops.add(full_13, uniform_4)
        del uniform_4

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_32)
        del add_32

        # pd_op.divide: (64x784x192xf32) <- (64x784x192xf32, xf32)
        divide_4 = paddle._C_ops.divide(reshape_61, full_13)

        # pd_op.multiply: (64x784x192xf32) <- (64x784x192xf32, 64x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 64x784x192xf32)
        add_33 = paddle._C_ops.add(add_27, multiply_4)

        # pd_op.layer_norm: (64x784x192xf32, 64x784xf32, 64x784xf32) <- (64x784x192xf32, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_33, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.matmul: (64x784x768xf32) <- (64x784x192xf32, 192x768xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_27, parameter_109, False, False)
        del parameter_109

        # pd_op.add: (64x784x768xf32) <- (64x784x768xf32, 768xf32)
        add_34 = paddle._C_ops.add(matmul_23, parameter_108)
        del parameter_108

        # pd_op.gelu: (64x784x768xf32) <- (64x784x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_34, False)

        # pd_op.matmul: (64x784x192xf32) <- (64x784x768xf32, 768x192xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 192xf32)
        add_35 = paddle._C_ops.add(matmul_24, parameter_106)
        del parameter_106

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_36 = paddle._C_ops.add(full_13, uniform_5)
        del uniform_5

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_36)
        del add_36

        # pd_op.divide: (64x784x192xf32) <- (64x784x192xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_35, full_13)

        # pd_op.multiply: (64x784x192xf32) <- (64x784x192xf32, 64x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (64x784x192xf32) <- (64x784x192xf32, 64x784x192xf32)
        add_37 = paddle._C_ops.add(add_33, multiply_5)

        # pd_op.reshape: (64x28x28x192xf32) <- (64x784x192xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(add_37, full_int_array_42)

        # pd_op.strided_slice: (64x14x14x192xf32) <- (64x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_17, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x14x14x192xf32) <- (64x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_39, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x14x14x192xf32) <- (64x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_40, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x14x14x192xf32) <- (64x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_19, full_int_array_30, full_int_array_38
        )

        # pd_op.reshape: (64x28x28x192xf32) <- (64x28x28x192xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(reshape_62, full_int_array_42)
        del full_int_array_42

        # builtin.combine: ([64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32]) <- (64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32)
        combine_1 = [strided_slice_4, strided_slice_5, strided_slice_6, strided_slice_7]

        # pd_op.concat: (64x14x14x768xf32) <- ([64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32, 64x14x14x192xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_8)
        del combine_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_54 = [64, -1, 768]

        # pd_op.reshape: (64x196x768xf32) <- (64x14x14x768xf32, 3xi64)
        reshape_64 = paddle._C_ops.reshape(concat_1, full_int_array_54)
        del full_int_array_54

        # pd_op.layer_norm: (64x196x768xf32, 64x196xf32, 64x196xf32) <- (64x196x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_64, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105

        # pd_op.matmul: (64x196x384xf32) <- (64x196x768xf32, 768x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_30, parameter_103, False, False)
        del parameter_103

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_25, parameter_102, parameter_101, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_101, parameter_102

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [64, 14, 14, 384]

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_33, full_int_array_55)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_56 = [64, 2, 7, 2, 7, 384]

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_66 = paddle._C_ops.reshape(reshape_65, full_int_array_56)

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_66, [0, 1, 3, 2, 4, 5])
        del reshape_66

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [-1, 7, 7, 384]

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_27, full_int_array_57)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_58 = [-1, 49, 384]

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_68 = paddle._C_ops.reshape(reshape_67, full_int_array_58)

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_68, parameter_100, False, False)
        del parameter_100

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_38 = paddle._C_ops.add(matmul_26, parameter_99)
        del parameter_99

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_59 = [256, 49, 3, 12, 32]

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_69 = paddle._C_ops.reshape(add_38, full_int_array_59)

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_69, [2, 0, 3, 1, 4])
        del reshape_69

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_12, full_0, float("0"), True)
        del slice_12

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_13, [0, 1, 3, 2])
        del slice_13

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, transpose_29, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_70 = paddle._C_ops.reshape(data_16, full_int_array_10)
        del data_16

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_4 = paddle._C_ops.index_select(data_6, reshape_70, 0)
        del data_6

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(index_select_4, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_71, [2, 0, 1])
        del reshape_71

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(transpose_30, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_39 = paddle._C_ops.add(matmul_27, unsqueeze_12)

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_4 = paddle._C_ops.softmax(add_39, -1)
        del add_39

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_28 = paddle._C_ops.matmul(softmax_4, slice_14, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_60 = [256, 49, 384]

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_72 = paddle._C_ops.reshape(transpose_31, full_int_array_60)

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_72, parameter_98, False, False)
        del parameter_98

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_40 = paddle._C_ops.add(matmul_29, parameter_97)
        del parameter_97

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_40, full_int_array_57)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_61 = [-1, 2, 2, 7, 7, 384]

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_74 = paddle._C_ops.reshape(reshape_73, full_int_array_61)

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_74, [0, 1, 3, 2, 4, 5])
        del reshape_74

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [-1, 14, 14, 384]

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_32, full_int_array_62)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_63 = [64, 196, 384]

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(reshape_75, full_int_array_63)

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.927273"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_142 = full_14

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_41 = paddle._C_ops.add(full_14, uniform_6)
        del uniform_6

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_41)
        del add_41

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_6 = paddle._C_ops.divide(reshape_76, full_14)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_42 = paddle._C_ops.add(matmul_25, multiply_6)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_42, parameter_96, parameter_95, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_95, parameter_96

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_36, parameter_94, False, False)
        del parameter_94

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_43 = paddle._C_ops.add(matmul_30, parameter_93)
        del parameter_93

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_43, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_4, parameter_92, False, False)
        del parameter_92

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_44 = paddle._C_ops.add(matmul_31, parameter_91)
        del parameter_91

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_45 = paddle._C_ops.add(full_14, uniform_7)
        del uniform_7

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_45)
        del add_45

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_7 = paddle._C_ops.divide(add_44, full_14)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_46 = paddle._C_ops.add(add_42, multiply_7)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_90, parameter_89, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_89, parameter_90

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(layer_norm_39, full_int_array_55)

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_4 = paddle._C_ops.roll(reshape_77, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_78 = paddle._C_ops.reshape(roll_4, full_int_array_56)

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_78, [0, 1, 3, 2, 4, 5])
        del reshape_78

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(transpose_33, full_int_array_57)

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_80 = paddle._C_ops.reshape(reshape_79, full_int_array_58)

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_15 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__18 = paddle._C_ops.set_value_(
            full_15,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_15

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__19 = paddle._C_ops.set_value_(
            set_value__18,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__18

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__20 = paddle._C_ops.set_value_(
            set_value__19,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__19

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__21 = paddle._C_ops.set_value_(
            set_value__20,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__20

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__22 = paddle._C_ops.set_value_(
            set_value__21,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__21

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__23 = paddle._C_ops.set_value_(
            set_value__22,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__22

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__24 = paddle._C_ops.set_value_(
            set_value__23,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__23

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__25 = paddle._C_ops.set_value_(
            set_value__24,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__24

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__26 = paddle._C_ops.set_value_(
            set_value__25,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__25

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_64 = [1, 2, 7, 2, 7, 1]

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_81 = paddle._C_ops.reshape(set_value__26, full_int_array_64)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_81, [0, 1, 3, 2, 4, 5])
        del reshape_81

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(transpose_34, full_int_array_32)
        del transpose_34

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_83 = paddle._C_ops.reshape(reshape_82, full_int_array_33)
        del reshape_82

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(reshape_83, full_int_array_7)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(reshape_83, full_int_array_8)
        del reshape_83

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_2 = paddle._C_ops.subtract(unsqueeze_13, unsqueeze_14)
        del unsqueeze_13, unsqueeze_14

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_2 = paddle._C_ops.not_equal(subtract_2, full_2)

        # pd_op.full: (4x49x49xf32) <- ()
        full_16 = paddle._C_ops.full(
            [4, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_4 = paddle._C_ops.where(not_equal_2, full_16, subtract_2)
        del not_equal_2, subtract_2

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_2 = paddle._C_ops.equal(where_4, full_2)

        # pd_op.full: (4x49x49xf32) <- ()
        full_17 = paddle._C_ops.full(
            [4, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_5 = paddle._C_ops.where(equal_2, full_17, where_4)
        del equal_2, where_4

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_80, parameter_88, False, False)
        del parameter_88

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_47 = paddle._C_ops.add(matmul_32, parameter_87)
        del parameter_87

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_84 = paddle._C_ops.reshape(add_47, full_int_array_59)

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_84, [2, 0, 3, 1, 4])
        del reshape_84

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_15, full_0, float("0"), True)
        del slice_15

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_36 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])
        del slice_16

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_33 = paddle._C_ops.matmul(scale_5, transpose_36, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_85 = paddle._C_ops.reshape(data_17, full_int_array_10)
        del data_17

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_5 = paddle._C_ops.index_select(data_7, reshape_85, 0)
        del data_7

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_86 = paddle._C_ops.reshape(index_select_5, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_86, [2, 0, 1])
        del reshape_86

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(transpose_37, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_48 = paddle._C_ops.add(matmul_33, unsqueeze_15)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_65 = [64, 4, 12, 49, 49]

        # pd_op.reshape: (64x4x12x49x49xf32) <- (256x12x49x49xf32, 5xi64)
        reshape_87 = paddle._C_ops.reshape(add_48, full_int_array_65)

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(where_5, full_int_array_7)
        del where_5

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(unsqueeze_16, full_int_array_6)
        del unsqueeze_16

        # pd_op.add: (64x4x12x49x49xf32) <- (64x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_49 = paddle._C_ops.add(reshape_87, unsqueeze_17)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [256, 12, 49, 49]

        # pd_op.reshape: (256x12x49x49xf32) <- (64x4x12x49x49xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(add_49, full_int_array_66)

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_88, -1)
        del reshape_88

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_34 = paddle._C_ops.matmul(softmax_5, slice_17, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])
        del matmul_34

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_89 = paddle._C_ops.reshape(transpose_38, full_int_array_60)

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_35 = paddle._C_ops.matmul(reshape_89, parameter_86, False, False)
        del parameter_86

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_50 = paddle._C_ops.add(matmul_35, parameter_85)
        del parameter_85

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(add_50, full_int_array_57)

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_91 = paddle._C_ops.reshape(reshape_90, full_int_array_61)

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_91, [0, 1, 3, 2, 4, 5])
        del reshape_91

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(transpose_39, full_int_array_62)

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape_92, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_93 = paddle._C_ops.reshape(roll_5, full_int_array_63)

        # pd_op.full: (xf32) <- ()
        full_18 = paddle._C_ops.full(
            [],
            float("0.909091"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_143 = full_18

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_51 = paddle._C_ops.add(full_18, uniform_8)
        del uniform_8

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_51)
        del add_51

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_8 = paddle._C_ops.divide(reshape_93, full_18)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_52 = paddle._C_ops.add(add_46, multiply_8)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_52, parameter_84, parameter_83, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_83, parameter_84

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_42, parameter_82, False, False)
        del parameter_82

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_53 = paddle._C_ops.add(matmul_36, parameter_81)
        del parameter_81

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_53, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_37 = paddle._C_ops.matmul(gelu_5, parameter_80, False, False)
        del parameter_80

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_54 = paddle._C_ops.add(matmul_37, parameter_79)
        del parameter_79

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_55 = paddle._C_ops.add(full_18, uniform_9)
        del uniform_9

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_55)
        del add_55

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_54, full_18)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_56 = paddle._C_ops.add(add_52, multiply_9)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_78, parameter_77, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_77, parameter_78

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(layer_norm_45, full_int_array_55)

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_95 = paddle._C_ops.reshape(reshape_94, full_int_array_56)

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_95, [0, 1, 3, 2, 4, 5])
        del reshape_95

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_96 = paddle._C_ops.reshape(transpose_40, full_int_array_57)

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_97 = paddle._C_ops.reshape(reshape_96, full_int_array_58)

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_97, parameter_76, False, False)
        del parameter_76

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_57 = paddle._C_ops.add(matmul_38, parameter_75)
        del parameter_75

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_98 = paddle._C_ops.reshape(add_57, full_int_array_59)

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_98, [2, 0, 3, 1, 4])
        del reshape_98

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_18, full_0, float("0"), True)
        del slice_18

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_42 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])
        del slice_19

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_39 = paddle._C_ops.matmul(scale_6, transpose_42, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_99 = paddle._C_ops.reshape(data_18, full_int_array_10)
        del data_18

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_6 = paddle._C_ops.index_select(data_8, reshape_99, 0)
        del data_8

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_100 = paddle._C_ops.reshape(index_select_6, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_100, [2, 0, 1])
        del reshape_100

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(transpose_43, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_58 = paddle._C_ops.add(matmul_39, unsqueeze_18)

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_6 = paddle._C_ops.softmax(add_58, -1)
        del add_58

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_40 = paddle._C_ops.matmul(softmax_6, slice_20, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_44 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_101 = paddle._C_ops.reshape(transpose_44, full_int_array_60)

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_101, parameter_74, False, False)
        del parameter_74

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_59 = paddle._C_ops.add(matmul_41, parameter_73)
        del parameter_73

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_102 = paddle._C_ops.reshape(add_59, full_int_array_57)

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_103 = paddle._C_ops.reshape(reshape_102, full_int_array_61)

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_103, [0, 1, 3, 2, 4, 5])
        del reshape_103

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_104 = paddle._C_ops.reshape(transpose_45, full_int_array_62)

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_105 = paddle._C_ops.reshape(reshape_104, full_int_array_63)

        # pd_op.full: (xf32) <- ()
        full_19 = paddle._C_ops.full(
            [],
            float("0.890909"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_144 = full_19

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_60 = paddle._C_ops.add(full_19, uniform_10)
        del uniform_10

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_60)
        del add_60

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_10 = paddle._C_ops.divide(reshape_105, full_19)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_61 = paddle._C_ops.add(add_56, multiply_10)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_72, parameter_71, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_71, parameter_72

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_48, parameter_70, False, False)
        del parameter_70

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_62 = paddle._C_ops.add(matmul_42, parameter_69)
        del parameter_69

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_62, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_43 = paddle._C_ops.matmul(gelu_6, parameter_68, False, False)
        del parameter_68

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_63 = paddle._C_ops.add(matmul_43, parameter_67)
        del parameter_67

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_64 = paddle._C_ops.add(full_19, uniform_11)
        del uniform_11

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_64)
        del add_64

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_11 = paddle._C_ops.divide(add_63, full_19)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_65 = paddle._C_ops.add(add_61, multiply_11)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_66, parameter_65, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_65, parameter_66

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_106 = paddle._C_ops.reshape(layer_norm_51, full_int_array_55)

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_6 = paddle._C_ops.roll(reshape_106, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_107 = paddle._C_ops.reshape(roll_6, full_int_array_56)

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_107, [0, 1, 3, 2, 4, 5])
        del reshape_107

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_108 = paddle._C_ops.reshape(transpose_46, full_int_array_57)

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_109 = paddle._C_ops.reshape(reshape_108, full_int_array_58)

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_20 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__27 = paddle._C_ops.set_value_(
            full_20,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_20

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__28 = paddle._C_ops.set_value_(
            set_value__27,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__27

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__29 = paddle._C_ops.set_value_(
            set_value__28,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__28

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__30 = paddle._C_ops.set_value_(
            set_value__29,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__29

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__31 = paddle._C_ops.set_value_(
            set_value__30,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__30

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__32 = paddle._C_ops.set_value_(
            set_value__31,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__31

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__33 = paddle._C_ops.set_value_(
            set_value__32,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__32

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__34 = paddle._C_ops.set_value_(
            set_value__33,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__33

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__35 = paddle._C_ops.set_value_(
            set_value__34,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__34

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_110 = paddle._C_ops.reshape(set_value__35, full_int_array_64)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_110, [0, 1, 3, 2, 4, 5])
        del reshape_110

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_111 = paddle._C_ops.reshape(transpose_47, full_int_array_32)
        del transpose_47

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_112 = paddle._C_ops.reshape(reshape_111, full_int_array_33)
        del reshape_111

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(reshape_112, full_int_array_7)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(reshape_112, full_int_array_8)
        del reshape_112

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_3 = paddle._C_ops.subtract(unsqueeze_19, unsqueeze_20)
        del unsqueeze_19, unsqueeze_20

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_3 = paddle._C_ops.not_equal(subtract_3, full_2)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_6 = paddle._C_ops.where(not_equal_3, full_16, subtract_3)
        del not_equal_3, subtract_3

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_3 = paddle._C_ops.equal(where_6, full_2)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_7 = paddle._C_ops.where(equal_3, full_17, where_6)
        del equal_3, where_6

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_44 = paddle._C_ops.matmul(reshape_109, parameter_64, False, False)
        del parameter_64

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_66 = paddle._C_ops.add(matmul_44, parameter_63)
        del parameter_63

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_113 = paddle._C_ops.reshape(add_66, full_int_array_59)

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_113, [2, 0, 3, 1, 4])
        del reshape_113

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_21, full_0, float("0"), True)
        del slice_21

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_49 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])
        del slice_22

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_45 = paddle._C_ops.matmul(scale_7, transpose_49, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_114 = paddle._C_ops.reshape(data_19, full_int_array_10)
        del data_19

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_7 = paddle._C_ops.index_select(data_9, reshape_114, 0)
        del data_9

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_115 = paddle._C_ops.reshape(index_select_7, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_115, [2, 0, 1])
        del reshape_115

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(transpose_50, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_67 = paddle._C_ops.add(matmul_45, unsqueeze_21)

        # pd_op.reshape: (64x4x12x49x49xf32) <- (256x12x49x49xf32, 5xi64)
        reshape_116 = paddle._C_ops.reshape(add_67, full_int_array_65)

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(where_7, full_int_array_7)
        del where_7

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(unsqueeze_22, full_int_array_6)
        del unsqueeze_22

        # pd_op.add: (64x4x12x49x49xf32) <- (64x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_68 = paddle._C_ops.add(reshape_116, unsqueeze_23)

        # pd_op.reshape: (256x12x49x49xf32) <- (64x4x12x49x49xf32, 4xi64)
        reshape_117 = paddle._C_ops.reshape(add_68, full_int_array_66)

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_117, -1)
        del reshape_117

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_46 = paddle._C_ops.matmul(softmax_7, slice_23, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_118 = paddle._C_ops.reshape(transpose_51, full_int_array_60)

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_118, parameter_62, False, False)
        del parameter_62

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_69 = paddle._C_ops.add(matmul_47, parameter_61)
        del parameter_61

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_119 = paddle._C_ops.reshape(add_69, full_int_array_57)

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_120 = paddle._C_ops.reshape(reshape_119, full_int_array_61)

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_120, [0, 1, 3, 2, 4, 5])
        del reshape_120

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_121 = paddle._C_ops.reshape(transpose_52, full_int_array_62)

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_7 = paddle._C_ops.roll(reshape_121, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_122 = paddle._C_ops.reshape(roll_7, full_int_array_63)

        # pd_op.full: (xf32) <- ()
        full_21 = paddle._C_ops.full(
            [],
            float("0.872727"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_145 = full_21

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_70 = paddle._C_ops.add(full_21, uniform_12)
        del uniform_12

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_70)
        del add_70

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_12 = paddle._C_ops.divide(reshape_122, full_21)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_71 = paddle._C_ops.add(add_65, multiply_12)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_71, parameter_60, parameter_59, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_59, parameter_60

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_54, parameter_58, False, False)
        del parameter_58

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_72 = paddle._C_ops.add(matmul_48, parameter_57)
        del parameter_57

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_72, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_7, parameter_56, False, False)
        del parameter_56

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_73 = paddle._C_ops.add(matmul_49, parameter_55)
        del parameter_55

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_74 = paddle._C_ops.add(full_21, uniform_13)
        del uniform_13

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_74)
        del add_74

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_73, full_21)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_75 = paddle._C_ops.add(add_71, multiply_13)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_75, parameter_54, parameter_53, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_53, parameter_54

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_123 = paddle._C_ops.reshape(layer_norm_57, full_int_array_55)

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_124 = paddle._C_ops.reshape(reshape_123, full_int_array_56)

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_124, [0, 1, 3, 2, 4, 5])
        del reshape_124

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_125 = paddle._C_ops.reshape(transpose_53, full_int_array_57)

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_126 = paddle._C_ops.reshape(reshape_125, full_int_array_58)

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_50 = paddle._C_ops.matmul(reshape_126, parameter_52, False, False)
        del parameter_52

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_76 = paddle._C_ops.add(matmul_50, parameter_51)
        del parameter_51

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_127 = paddle._C_ops.reshape(add_76, full_int_array_59)

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_127, [2, 0, 3, 1, 4])
        del reshape_127

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_24, full_0, float("0"), True)
        del slice_24

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_55 = paddle._C_ops.transpose(slice_25, [0, 1, 3, 2])
        del slice_25

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_51 = paddle._C_ops.matmul(scale_8, transpose_55, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_128 = paddle._C_ops.reshape(data_20, full_int_array_10)
        del data_20

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_8 = paddle._C_ops.index_select(data_10, reshape_128, 0)
        del data_10

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_129 = paddle._C_ops.reshape(index_select_8, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_129, [2, 0, 1])
        del reshape_129

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(transpose_56, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_77 = paddle._C_ops.add(matmul_51, unsqueeze_24)

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_8 = paddle._C_ops.softmax(add_77, -1)
        del add_77

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_8, slice_26, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_130 = paddle._C_ops.reshape(transpose_57, full_int_array_60)

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_130, parameter_50, False, False)
        del parameter_50

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_78 = paddle._C_ops.add(matmul_53, parameter_49)
        del parameter_49

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_131 = paddle._C_ops.reshape(add_78, full_int_array_57)

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_132 = paddle._C_ops.reshape(reshape_131, full_int_array_61)

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_132, [0, 1, 3, 2, 4, 5])
        del reshape_132

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_133 = paddle._C_ops.reshape(transpose_58, full_int_array_62)

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_134 = paddle._C_ops.reshape(reshape_133, full_int_array_63)

        # pd_op.full: (xf32) <- ()
        full_22 = paddle._C_ops.full(
            [],
            float("0.854545"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_146 = full_22

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_79 = paddle._C_ops.add(full_22, uniform_14)
        del uniform_14

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_79)
        del add_79

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_14 = paddle._C_ops.divide(reshape_134, full_22)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_80 = paddle._C_ops.add(add_75, multiply_14)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_48, parameter_47, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_60, parameter_46, False, False)
        del parameter_46

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_81 = paddle._C_ops.add(matmul_54, parameter_45)
        del parameter_45

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_81, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_8, parameter_44, False, False)
        del parameter_44

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_82 = paddle._C_ops.add(matmul_55, parameter_43)
        del parameter_43

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_83 = paddle._C_ops.add(full_22, uniform_15)
        del uniform_15

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_83)
        del add_83

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_15 = paddle._C_ops.divide(add_82, full_22)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_84 = paddle._C_ops.add(add_80, multiply_15)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_84, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_135 = paddle._C_ops.reshape(layer_norm_63, full_int_array_55)

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_8 = paddle._C_ops.roll(reshape_135, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x2x7x2x7x384xf32) <- (64x14x14x384xf32, 6xi64)
        reshape_136 = paddle._C_ops.reshape(roll_8, full_int_array_56)
        del full_int_array_56

        # pd_op.transpose: (64x2x2x7x7x384xf32) <- (64x2x7x2x7x384xf32)
        transpose_59 = paddle._C_ops.transpose(reshape_136, [0, 1, 3, 2, 4, 5])
        del reshape_136

        # pd_op.reshape: (256x7x7x384xf32) <- (64x2x2x7x7x384xf32, 4xi64)
        reshape_137 = paddle._C_ops.reshape(transpose_59, full_int_array_57)

        # pd_op.reshape: (256x49x384xf32) <- (256x7x7x384xf32, 3xi64)
        reshape_138 = paddle._C_ops.reshape(reshape_137, full_int_array_58)
        del full_int_array_58

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_23 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__36 = paddle._C_ops.set_value_(
            full_23,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_23

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__37 = paddle._C_ops.set_value_(
            set_value__36,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__36

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__38 = paddle._C_ops.set_value_(
            set_value__37,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__37

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__39 = paddle._C_ops.set_value_(
            set_value__38,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del set_value__38

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__40 = paddle._C_ops.set_value_(
            set_value__39,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__39

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__41 = paddle._C_ops.set_value_(
            set_value__40,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__40

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__42 = paddle._C_ops.set_value_(
            set_value__41,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__41

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__43 = paddle._C_ops.set_value_(
            set_value__42,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__42

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__44 = paddle._C_ops.set_value_(
            set_value__43,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__43

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_139 = paddle._C_ops.reshape(set_value__44, full_int_array_64)
        del full_int_array_64

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_139, [0, 1, 3, 2, 4, 5])
        del reshape_139

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_140 = paddle._C_ops.reshape(transpose_60, full_int_array_32)
        del transpose_60

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_141 = paddle._C_ops.reshape(reshape_140, full_int_array_33)
        del reshape_140

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(reshape_141, full_int_array_7)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(reshape_141, full_int_array_8)
        del reshape_141

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_4 = paddle._C_ops.subtract(unsqueeze_25, unsqueeze_26)
        del unsqueeze_25, unsqueeze_26

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_4 = paddle._C_ops.not_equal(subtract_4, full_2)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_8 = paddle._C_ops.where(not_equal_4, full_16, subtract_4)
        del full_16, not_equal_4, subtract_4

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_4 = paddle._C_ops.equal(where_8, full_2)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_9 = paddle._C_ops.where(equal_4, full_17, where_8)
        del equal_4, full_17, where_8

        # pd_op.matmul: (256x49x1152xf32) <- (256x49x384xf32, 384x1152xf32)
        matmul_56 = paddle._C_ops.matmul(reshape_138, parameter_40, False, False)
        del parameter_40

        # pd_op.add: (256x49x1152xf32) <- (256x49x1152xf32, 1152xf32)
        add_85 = paddle._C_ops.add(matmul_56, parameter_39)
        del parameter_39

        # pd_op.reshape: (256x49x3x12x32xf32) <- (256x49x1152xf32, 5xi64)
        reshape_142 = paddle._C_ops.reshape(add_85, full_int_array_59)
        del full_int_array_59

        # pd_op.transpose: (3x256x12x49x32xf32) <- (256x49x3x12x32xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_142, [2, 0, 3, 1, 4])
        del reshape_142

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (256x12x49x32xf32) <- (3x256x12x49x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (256x12x49x32xf32) <- (256x12x49x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_27, full_0, float("0"), True)
        del slice_27

        # pd_op.transpose: (256x12x32x49xf32) <- (256x12x49x32xf32)
        transpose_62 = paddle._C_ops.transpose(slice_28, [0, 1, 3, 2])
        del slice_28

        # pd_op.matmul: (256x12x49x49xf32) <- (256x12x49x32xf32, 256x12x32x49xf32)
        matmul_57 = paddle._C_ops.matmul(scale_9, transpose_62, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_143 = paddle._C_ops.reshape(data_21, full_int_array_10)
        del data_21

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_9 = paddle._C_ops.index_select(data_11, reshape_143, 0)
        del data_11

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_144 = paddle._C_ops.reshape(index_select_9, full_int_array_11)

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_144, [2, 0, 1])
        del reshape_144

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_6)

        # pd_op.add: (256x12x49x49xf32) <- (256x12x49x49xf32, 1x12x49x49xf32)
        add_86 = paddle._C_ops.add(matmul_57, unsqueeze_27)

        # pd_op.reshape: (64x4x12x49x49xf32) <- (256x12x49x49xf32, 5xi64)
        reshape_145 = paddle._C_ops.reshape(add_86, full_int_array_65)
        del full_int_array_65

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(where_9, full_int_array_7)
        del where_9

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(unsqueeze_28, full_int_array_6)
        del unsqueeze_28

        # pd_op.add: (64x4x12x49x49xf32) <- (64x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_87 = paddle._C_ops.add(reshape_145, unsqueeze_29)

        # pd_op.reshape: (256x12x49x49xf32) <- (64x4x12x49x49xf32, 4xi64)
        reshape_146 = paddle._C_ops.reshape(add_87, full_int_array_66)
        del full_int_array_66

        # pd_op.softmax: (256x12x49x49xf32) <- (256x12x49x49xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_146, -1)
        del reshape_146

        # pd_op.matmul: (256x12x49x32xf32) <- (256x12x49x49xf32, 256x12x49x32xf32)
        matmul_58 = paddle._C_ops.matmul(softmax_9, slice_29, False, False)

        # pd_op.transpose: (256x49x12x32xf32) <- (256x12x49x32xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])
        del matmul_58

        # pd_op.reshape: (256x49x384xf32) <- (256x49x12x32xf32, 3xi64)
        reshape_147 = paddle._C_ops.reshape(transpose_64, full_int_array_60)
        del full_int_array_60

        # pd_op.matmul: (256x49x384xf32) <- (256x49x384xf32, 384x384xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_147, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (256x49x384xf32) <- (256x49x384xf32, 384xf32)
        add_88 = paddle._C_ops.add(matmul_59, parameter_37)
        del parameter_37

        # pd_op.reshape: (256x7x7x384xf32) <- (256x49x384xf32, 4xi64)
        reshape_148 = paddle._C_ops.reshape(add_88, full_int_array_57)
        del full_int_array_57

        # pd_op.reshape: (64x2x2x7x7x384xf32) <- (256x7x7x384xf32, 6xi64)
        reshape_149 = paddle._C_ops.reshape(reshape_148, full_int_array_61)
        del full_int_array_61

        # pd_op.transpose: (64x2x7x2x7x384xf32) <- (64x2x2x7x7x384xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_149, [0, 1, 3, 2, 4, 5])
        del reshape_149

        # pd_op.reshape: (64x14x14x384xf32) <- (64x2x7x2x7x384xf32, 4xi64)
        reshape_150 = paddle._C_ops.reshape(transpose_65, full_int_array_62)
        del full_int_array_62

        # pd_op.roll: (64x14x14x384xf32) <- (64x14x14x384xf32, 2xi64)
        roll_9 = paddle._C_ops.roll(reshape_150, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x196x384xf32) <- (64x14x14x384xf32, 3xi64)
        reshape_151 = paddle._C_ops.reshape(roll_9, full_int_array_63)
        del full_int_array_63

        # pd_op.full: (xf32) <- ()
        full_24 = paddle._C_ops.full(
            [],
            float("0.836364"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_147 = full_24

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_89 = paddle._C_ops.add(full_24, uniform_16)
        del uniform_16

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_89)
        del add_89

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_16 = paddle._C_ops.divide(reshape_151, full_24)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_90 = paddle._C_ops.add(add_84, multiply_16)

        # pd_op.layer_norm: (64x196x384xf32, 64x196xf32, 64x196xf32) <- (64x196x384xf32, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_90, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (64x196x1536xf32) <- (64x196x384xf32, 384x1536xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_66, parameter_34, False, False)
        del parameter_34

        # pd_op.add: (64x196x1536xf32) <- (64x196x1536xf32, 1536xf32)
        add_91 = paddle._C_ops.add(matmul_60, parameter_33)
        del parameter_33

        # pd_op.gelu: (64x196x1536xf32) <- (64x196x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_91, False)

        # pd_op.matmul: (64x196x384xf32) <- (64x196x1536xf32, 1536x384xf32)
        matmul_61 = paddle._C_ops.matmul(gelu_9, parameter_32, False, False)
        del parameter_32

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 384xf32)
        add_92 = paddle._C_ops.add(matmul_61, parameter_31)
        del parameter_31

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_17 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_93 = paddle._C_ops.add(full_24, uniform_17)
        del uniform_17

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_17 = paddle._C_ops.floor(add_93)
        del add_93

        # pd_op.divide: (64x196x384xf32) <- (64x196x384xf32, xf32)
        divide_17 = paddle._C_ops.divide(add_92, full_24)

        # pd_op.multiply: (64x196x384xf32) <- (64x196x384xf32, 64x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_17, floor_17)

        # pd_op.add: (64x196x384xf32) <- (64x196x384xf32, 64x196x384xf32)
        add_94 = paddle._C_ops.add(add_90, multiply_17)

        # pd_op.reshape: (64x14x14x384xf32) <- (64x196x384xf32, 4xi64)
        reshape_152 = paddle._C_ops.reshape(add_94, full_int_array_55)

        # pd_op.strided_slice: (64x7x7x384xf32) <- (64x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_17, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x7x7x384xf32) <- (64x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_39, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x7x7x384xf32) <- (64x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_40, full_int_array_30, full_int_array_38
        )

        # pd_op.strided_slice: (64x7x7x384xf32) <- (64x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_19, full_int_array_30, full_int_array_38
        )

        # pd_op.reshape: (64x14x14x384xf32) <- (64x14x14x384xf32, 4xi64)
        reshape_153 = paddle._C_ops.reshape(reshape_152, full_int_array_55)
        del full_int_array_55

        # builtin.combine: ([64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32]) <- (64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32)
        combine_2 = [
            strided_slice_8,
            strided_slice_9,
            strided_slice_10,
            strided_slice_11,
        ]

        # pd_op.concat: (64x7x7x1536xf32) <- ([64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32, 64x7x7x384xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_8)
        del combine_2

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_67 = [64, -1, 1536]

        # pd_op.reshape: (64x49x1536xf32) <- (64x7x7x1536xf32, 3xi64)
        reshape_154 = paddle._C_ops.reshape(concat_2, full_int_array_67)
        del full_int_array_67

        # pd_op.layer_norm: (64x49x1536xf32, 64x49xf32, 64x49xf32) <- (64x49x1536xf32, 1536xf32, 1536xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_154, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30

        # pd_op.matmul: (64x49x768xf32) <- (64x49x1536xf32, 1536x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_69, parameter_28, False, False)
        del parameter_28

        # pd_op.layer_norm: (64x49x768xf32, 64x49xf32, 64x49xf32) <- (64x49x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_62, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [64, 7, 7, 768]

        # pd_op.reshape: (64x7x7x768xf32) <- (64x49x768xf32, 4xi64)
        reshape_155 = paddle._C_ops.reshape(layer_norm_72, full_int_array_68)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_69 = [64, 1, 7, 1, 7, 768]

        # pd_op.reshape: (64x1x7x1x7x768xf32) <- (64x7x7x768xf32, 6xi64)
        reshape_156 = paddle._C_ops.reshape(reshape_155, full_int_array_69)

        # pd_op.transpose: (64x1x1x7x7x768xf32) <- (64x1x7x1x7x768xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_156, [0, 1, 3, 2, 4, 5])
        del reshape_156

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_70 = [-1, 7, 7, 768]

        # pd_op.reshape: (64x7x7x768xf32) <- (64x1x1x7x7x768xf32, 4xi64)
        reshape_157 = paddle._C_ops.reshape(transpose_66, full_int_array_70)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_71 = [-1, 49, 768]

        # pd_op.reshape: (64x49x768xf32) <- (64x7x7x768xf32, 3xi64)
        reshape_158 = paddle._C_ops.reshape(reshape_157, full_int_array_71)

        # pd_op.matmul: (64x49x2304xf32) <- (64x49x768xf32, 768x2304xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_158, parameter_25, False, False)
        del parameter_25

        # pd_op.add: (64x49x2304xf32) <- (64x49x2304xf32, 2304xf32)
        add_95 = paddle._C_ops.add(matmul_63, parameter_24)
        del parameter_24

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_72 = [64, 49, 3, 24, 32]

        # pd_op.reshape: (64x49x3x24x32xf32) <- (64x49x2304xf32, 5xi64)
        reshape_159 = paddle._C_ops.reshape(add_95, full_int_array_72)

        # pd_op.transpose: (3x64x24x49x32xf32) <- (64x49x3x24x32xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_159, [2, 0, 3, 1, 4])
        del reshape_159

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (64x24x49x32xf32) <- (64x24x49x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_30, full_0, float("0"), True)
        del slice_30

        # pd_op.transpose: (64x24x32x49xf32) <- (64x24x49x32xf32)
        transpose_68 = paddle._C_ops.transpose(slice_31, [0, 1, 3, 2])
        del slice_31

        # pd_op.matmul: (64x24x49x49xf32) <- (64x24x49x32xf32, 64x24x32x49xf32)
        matmul_64 = paddle._C_ops.matmul(scale_10, transpose_68, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_160 = paddle._C_ops.reshape(data_22, full_int_array_10)
        del data_22

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_10 = paddle._C_ops.index_select(data_2, reshape_160, 0)
        del data_2

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_161 = paddle._C_ops.reshape(index_select_10, full_int_array_11)

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_161, [2, 0, 1])
        del reshape_161

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(transpose_69, full_int_array_6)

        # pd_op.add: (64x24x49x49xf32) <- (64x24x49x49xf32, 1x24x49x49xf32)
        add_96 = paddle._C_ops.add(matmul_64, unsqueeze_30)

        # pd_op.softmax: (64x24x49x49xf32) <- (64x24x49x49xf32)
        softmax_10 = paddle._C_ops.softmax(add_96, -1)
        del add_96

        # pd_op.matmul: (64x24x49x32xf32) <- (64x24x49x49xf32, 64x24x49x32xf32)
        matmul_65 = paddle._C_ops.matmul(softmax_10, slice_32, False, False)

        # pd_op.transpose: (64x49x24x32xf32) <- (64x24x49x32xf32)
        transpose_70 = paddle._C_ops.transpose(matmul_65, [0, 2, 1, 3])
        del matmul_65

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_73 = [64, 49, 768]

        # pd_op.reshape: (64x49x768xf32) <- (64x49x24x32xf32, 3xi64)
        reshape_162 = paddle._C_ops.reshape(transpose_70, full_int_array_73)

        # pd_op.matmul: (64x49x768xf32) <- (64x49x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(reshape_162, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 768xf32)
        add_97 = paddle._C_ops.add(matmul_66, parameter_22)
        del parameter_22

        # pd_op.reshape: (64x7x7x768xf32) <- (64x49x768xf32, 4xi64)
        reshape_163 = paddle._C_ops.reshape(add_97, full_int_array_70)

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_74 = [-1, 1, 1, 7, 7, 768]

        # pd_op.reshape: (64x1x1x7x7x768xf32) <- (64x7x7x768xf32, 6xi64)
        reshape_164 = paddle._C_ops.reshape(reshape_163, full_int_array_74)

        # pd_op.transpose: (64x1x7x1x7x768xf32) <- (64x1x1x7x7x768xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_164, [0, 1, 3, 2, 4, 5])
        del reshape_164

        # pd_op.reshape: (64x7x7x768xf32) <- (64x1x7x1x7x768xf32, 4xi64)
        reshape_165 = paddle._C_ops.reshape(transpose_71, full_int_array_70)

        # pd_op.reshape: (64x49x768xf32) <- (64x7x7x768xf32, 3xi64)
        reshape_166 = paddle._C_ops.reshape(reshape_165, full_int_array_73)

        # pd_op.full: (xf32) <- ()
        full_25 = paddle._C_ops.full(
            [],
            float("0.818182"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_148 = full_25

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_18 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_98 = paddle._C_ops.add(full_25, uniform_18)
        del uniform_18

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_18 = paddle._C_ops.floor(add_98)
        del add_98

        # pd_op.divide: (64x49x768xf32) <- (64x49x768xf32, xf32)
        divide_18 = paddle._C_ops.divide(reshape_166, full_25)

        # pd_op.multiply: (64x49x768xf32) <- (64x49x768xf32, 64x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_18, floor_18)

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 64x49x768xf32)
        add_99 = paddle._C_ops.add(matmul_62, multiply_18)

        # pd_op.layer_norm: (64x49x768xf32, 64x49xf32, 64x49xf32) <- (64x49x768xf32, 768xf32, 768xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_99, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (64x49x3072xf32) <- (64x49x768xf32, 768x3072xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_75, parameter_19, False, False)
        del parameter_19

        # pd_op.add: (64x49x3072xf32) <- (64x49x3072xf32, 3072xf32)
        add_100 = paddle._C_ops.add(matmul_67, parameter_18)
        del parameter_18

        # pd_op.gelu: (64x49x3072xf32) <- (64x49x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_100, False)

        # pd_op.matmul: (64x49x768xf32) <- (64x49x3072xf32, 3072x768xf32)
        matmul_68 = paddle._C_ops.matmul(gelu_10, parameter_17, False, False)
        del parameter_17

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_68, parameter_16)
        del parameter_16

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_19 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_102 = paddle._C_ops.add(full_25, uniform_19)
        del uniform_19

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_19 = paddle._C_ops.floor(add_102)
        del add_102

        # pd_op.divide: (64x49x768xf32) <- (64x49x768xf32, xf32)
        divide_19 = paddle._C_ops.divide(add_101, full_25)

        # pd_op.multiply: (64x49x768xf32) <- (64x49x768xf32, 64x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_19, floor_19)

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 64x49x768xf32)
        add_103 = paddle._C_ops.add(add_99, multiply_19)

        # pd_op.layer_norm: (64x49x768xf32, 64x49xf32, 64x49xf32) <- (64x49x768xf32, 768xf32, 768xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_103, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # pd_op.reshape: (64x7x7x768xf32) <- (64x49x768xf32, 4xi64)
        reshape_167 = paddle._C_ops.reshape(layer_norm_78, full_int_array_68)
        del full_int_array_68

        # pd_op.roll: (64x7x7x768xf32) <- (64x7x7x768xf32, 2xi64)
        roll_10 = paddle._C_ops.roll(reshape_167, full_int_array_16, [1, 2])

        # pd_op.reshape: (64x1x7x1x7x768xf32) <- (64x7x7x768xf32, 6xi64)
        reshape_168 = paddle._C_ops.reshape(roll_10, full_int_array_69)
        del full_int_array_69

        # pd_op.transpose: (64x1x1x7x7x768xf32) <- (64x1x7x1x7x768xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_168, [0, 1, 3, 2, 4, 5])
        del reshape_168

        # pd_op.reshape: (64x7x7x768xf32) <- (64x1x1x7x7x768xf32, 4xi64)
        reshape_169 = paddle._C_ops.reshape(transpose_72, full_int_array_70)

        # pd_op.reshape: (64x49x768xf32) <- (64x7x7x768xf32, 3xi64)
        reshape_170 = paddle._C_ops.reshape(reshape_169, full_int_array_71)
        del full_int_array_71

        # pd_op.full: (1x7x7x1xf32) <- ()
        full_26 = paddle._C_ops.full(
            [1, 7, 7, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__45 = paddle._C_ops.set_value_(
            full_26,
            full_int_array_17,
            full_int_array_18,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_26, full_int_array_17

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__46 = paddle._C_ops.set_value_(
            set_value__45,
            full_int_array_20,
            full_int_array_21,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_20, set_value__45

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__47 = paddle._C_ops.set_value_(
            set_value__46,
            full_int_array_22,
            full_int_array_23,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_22, full_int_array_23, set_value__46

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__48 = paddle._C_ops.set_value_(
            set_value__47,
            full_int_array_24,
            full_int_array_25,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del full_int_array_24, set_value__47

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__49 = paddle._C_ops.set_value_(
            set_value__48,
            full_int_array_18,
            full_int_array_16,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del full_int_array_18, set_value__48

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__50 = paddle._C_ops.set_value_(
            set_value__49,
            full_int_array_21,
            full_int_array_26,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del full_int_array_21, full_int_array_26, set_value__49

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__51 = paddle._C_ops.set_value_(
            set_value__50,
            full_int_array_27,
            full_int_array_28,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del full_int_array_27, full_int_array_28, set_value__50

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__52 = paddle._C_ops.set_value_(
            set_value__51,
            full_int_array_25,
            full_int_array_29,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del full_int_array_25, full_int_array_29, set_value__51

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__53 = paddle._C_ops.set_value_(
            set_value__52,
            full_int_array_16,
            full_int_array_30,
            full_int_array_19,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del full_int_array_30, set_value__52

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_75 = [1, 1, 7, 1, 7, 1]

        # pd_op.reshape: (1x1x7x1x7x1xf32) <- (1x7x7x1xf32, 6xi64)
        reshape_171 = paddle._C_ops.reshape(set_value__53, full_int_array_75)
        del full_int_array_75

        # pd_op.transpose: (1x1x1x7x7x1xf32) <- (1x1x7x1x7x1xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_171, [0, 1, 3, 2, 4, 5])
        del reshape_171

        # pd_op.reshape: (1x7x7x1xf32) <- (1x1x1x7x7x1xf32, 4xi64)
        reshape_172 = paddle._C_ops.reshape(transpose_73, full_int_array_32)
        del full_int_array_32, transpose_73

        # pd_op.reshape: (1x49xf32) <- (1x7x7x1xf32, 2xi64)
        reshape_173 = paddle._C_ops.reshape(reshape_172, full_int_array_33)
        del full_int_array_33, reshape_172

        # pd_op.unsqueeze: (1x1x49xf32) <- (1x49xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(reshape_173, full_int_array_7)

        # pd_op.unsqueeze: (1x49x1xf32) <- (1x49xf32, 1xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(reshape_173, full_int_array_8)
        del reshape_173

        # pd_op.subtract: (1x49x49xf32) <- (1x1x49xf32, 1x49x1xf32)
        subtract_5 = paddle._C_ops.subtract(unsqueeze_31, unsqueeze_32)
        del unsqueeze_31, unsqueeze_32

        # pd_op.not_equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        not_equal_5 = paddle._C_ops.not_equal(subtract_5, full_2)

        # pd_op.full: (1x49x49xf32) <- ()
        full_27 = paddle._C_ops.full(
            [1, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        where_10 = paddle._C_ops.where(not_equal_5, full_27, subtract_5)
        del full_27, not_equal_5, subtract_5

        # pd_op.equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        equal_5 = paddle._C_ops.equal(where_10, full_2)
        del full_2

        # pd_op.full: (1x49x49xf32) <- ()
        full_28 = paddle._C_ops.full(
            [1, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        where_11 = paddle._C_ops.where(equal_5, full_28, where_10)
        del equal_5, full_28, where_10

        # pd_op.matmul: (64x49x2304xf32) <- (64x49x768xf32, 768x2304xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_170, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (64x49x2304xf32) <- (64x49x2304xf32, 2304xf32)
        add_104 = paddle._C_ops.add(matmul_69, parameter_12)
        del parameter_12

        # pd_op.reshape: (64x49x3x24x32xf32) <- (64x49x2304xf32, 5xi64)
        reshape_174 = paddle._C_ops.reshape(add_104, full_int_array_72)
        del full_int_array_72

        # pd_op.transpose: (3x64x24x49x32xf32) <- (64x49x3x24x32xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_174, [2, 0, 3, 1, 4])
        del reshape_174

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_6, full_int_array_7, [1], [0]
        )

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (64x24x49x32xf32) <- (3x64x24x49x32xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.scale: (64x24x49x32xf32) <- (64x24x49x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_33, full_0, float("0"), True)
        del slice_33

        # pd_op.transpose: (64x24x32x49xf32) <- (64x24x49x32xf32)
        transpose_75 = paddle._C_ops.transpose(slice_34, [0, 1, 3, 2])
        del slice_34

        # pd_op.matmul: (64x24x49x49xf32) <- (64x24x49x32xf32, 64x24x32x49xf32)
        matmul_70 = paddle._C_ops.matmul(scale_11, transpose_75, False, False)

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_175 = paddle._C_ops.reshape(data_23, full_int_array_10)
        del data_23, full_int_array_10

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_11 = paddle._C_ops.index_select(data_3, reshape_175, 0)
        del data_3

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_176 = paddle._C_ops.reshape(index_select_11, full_int_array_11)
        del full_int_array_11

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_176, [2, 0, 1])
        del reshape_176

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(transpose_76, full_int_array_6)

        # pd_op.add: (64x24x49x49xf32) <- (64x24x49x49xf32, 1x24x49x49xf32)
        add_105 = paddle._C_ops.add(matmul_70, unsqueeze_33)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_76 = [64, 1, 24, 49, 49]

        # pd_op.reshape: (64x1x24x49x49xf32) <- (64x24x49x49xf32, 5xi64)
        reshape_177 = paddle._C_ops.reshape(add_105, full_int_array_76)
        del full_int_array_76

        # pd_op.unsqueeze: (1x1x49x49xf32) <- (1x49x49xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(where_11, full_int_array_7)
        del where_11

        # pd_op.unsqueeze: (1x1x1x49x49xf32) <- (1x1x49x49xf32, 1xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(unsqueeze_34, full_int_array_6)
        del unsqueeze_34

        # pd_op.add: (64x1x24x49x49xf32) <- (64x1x24x49x49xf32, 1x1x1x49x49xf32)
        add_106 = paddle._C_ops.add(reshape_177, unsqueeze_35)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_77 = [64, 24, 49, 49]

        # pd_op.reshape: (64x24x49x49xf32) <- (64x1x24x49x49xf32, 4xi64)
        reshape_178 = paddle._C_ops.reshape(add_106, full_int_array_77)
        del full_int_array_77

        # pd_op.softmax: (64x24x49x49xf32) <- (64x24x49x49xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_178, -1)
        del reshape_178

        # pd_op.matmul: (64x24x49x32xf32) <- (64x24x49x49xf32, 64x24x49x32xf32)
        matmul_71 = paddle._C_ops.matmul(softmax_11, slice_35, False, False)

        # pd_op.transpose: (64x49x24x32xf32) <- (64x24x49x32xf32)
        transpose_77 = paddle._C_ops.transpose(matmul_71, [0, 2, 1, 3])
        del matmul_71

        # pd_op.reshape: (64x49x768xf32) <- (64x49x24x32xf32, 3xi64)
        reshape_179 = paddle._C_ops.reshape(transpose_77, full_int_array_73)

        # pd_op.matmul: (64x49x768xf32) <- (64x49x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(reshape_179, parameter_11, False, False)
        del parameter_11

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 768xf32)
        add_107 = paddle._C_ops.add(matmul_72, parameter_10)
        del parameter_10

        # pd_op.reshape: (64x7x7x768xf32) <- (64x49x768xf32, 4xi64)
        reshape_180 = paddle._C_ops.reshape(add_107, full_int_array_70)

        # pd_op.reshape: (64x1x1x7x7x768xf32) <- (64x7x7x768xf32, 6xi64)
        reshape_181 = paddle._C_ops.reshape(reshape_180, full_int_array_74)
        del full_int_array_74

        # pd_op.transpose: (64x1x7x1x7x768xf32) <- (64x1x1x7x7x768xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_181, [0, 1, 3, 2, 4, 5])
        del reshape_181

        # pd_op.reshape: (64x7x7x768xf32) <- (64x1x7x1x7x768xf32, 4xi64)
        reshape_182 = paddle._C_ops.reshape(transpose_78, full_int_array_70)
        del full_int_array_70

        # pd_op.roll: (64x7x7x768xf32) <- (64x7x7x768xf32, 2xi64)
        roll_11 = paddle._C_ops.roll(reshape_182, full_int_array_36, [1, 2])

        # pd_op.reshape: (64x49x768xf32) <- (64x7x7x768xf32, 3xi64)
        reshape_183 = paddle._C_ops.reshape(roll_11, full_int_array_73)
        del full_int_array_73

        # pd_op.full: (xf32) <- ()
        full_29 = paddle._C_ops.full(
            [], float("0.8"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_149 = full_29

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_20 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_108 = paddle._C_ops.add(full_29, uniform_20)
        del uniform_20

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_20 = paddle._C_ops.floor(add_108)
        del add_108

        # pd_op.divide: (64x49x768xf32) <- (64x49x768xf32, xf32)
        divide_20 = paddle._C_ops.divide(reshape_183, full_29)

        # pd_op.multiply: (64x49x768xf32) <- (64x49x768xf32, 64x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_20, floor_20)

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 64x49x768xf32)
        add_109 = paddle._C_ops.add(add_103, multiply_20)

        # pd_op.layer_norm: (64x49x768xf32, 64x49xf32, 64x49xf32) <- (64x49x768xf32, 768xf32, 768xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_109, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (64x49x3072xf32) <- (64x49x768xf32, 768x3072xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_81, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (64x49x3072xf32) <- (64x49x3072xf32, 3072xf32)
        add_110 = paddle._C_ops.add(matmul_73, parameter_6)
        del parameter_6

        # pd_op.gelu: (64x49x3072xf32) <- (64x49x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_110, False)

        # pd_op.matmul: (64x49x768xf32) <- (64x49x3072xf32, 3072x768xf32)
        matmul_74 = paddle._C_ops.matmul(gelu_11, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 768xf32)
        add_111 = paddle._C_ops.add(matmul_74, parameter_4)
        del parameter_4

        # pd_op.uniform: (64x1x1xf32) <- (3xi64, 1xf32, 1xf32)
        uniform_21 = paddle._C_ops.uniform(
            full_int_array_37,
            paddle.float32,
            full_6,
            full_7,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_6, full_7, full_int_array_37

        # pd_op.add: (64x1x1xf32) <- (xf32, 64x1x1xf32)
        add_112 = paddle._C_ops.add(full_29, uniform_21)
        del uniform_21

        # pd_op.floor: (64x1x1xf32) <- (64x1x1xf32)
        floor_21 = paddle._C_ops.floor(add_112)
        del add_112

        # pd_op.divide: (64x49x768xf32) <- (64x49x768xf32, xf32)
        divide_21 = paddle._C_ops.divide(add_111, full_29)

        # pd_op.multiply: (64x49x768xf32) <- (64x49x768xf32, 64x1x1xf32)
        multiply_21 = paddle._C_ops.multiply(divide_21, floor_21)

        # pd_op.add: (64x49x768xf32) <- (64x49x768xf32, 64x49x768xf32)
        add_113 = paddle._C_ops.add(add_109, multiply_21)

        # pd_op.layer_norm: (64x49x768xf32, 64x49xf32, 64x49xf32) <- (64x49x768xf32, 768xf32, 768xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_113, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.transpose: (64x768x49xf32) <- (64x49x768xf32)
        transpose_79 = paddle._C_ops.transpose(layer_norm_84, [0, 2, 1])
        del layer_norm_84

        # pd_op.unsqueeze: (64x768x1x49xf32) <- (64x768x49xf32, 1xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(transpose_79, full_int_array_8)

        # pd_op.pool2d: (64x768x1x1xf32) <- (64x768x1x49xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            unsqueeze_36,
            full_int_array_19,
            [1, 1],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            True,
            "EXPLICIT",
        )
        del full_int_array_19

        # pd_op.squeeze: (64x768x1xf32) <- (64x768x1x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(pool2d_0, full_int_array_8)

        # pd_op.flatten: (64x768xf32) <- (64x768x1xf32)
        flatten_1 = paddle._C_ops.flatten(squeeze_0, 1, 2)

        # pd_op.matmul: (64x102xf32) <- (64x768xf32, 768x102xf32)
        matmul_75 = paddle._C_ops.matmul(flatten_1, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (64x102xf32) <- (64x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_75, parameter_0)
        del (
            add_1,
            add_10,
            add_100,
            add_101,
            add_103,
            add_104,
            add_105,
            add_106,
            add_107,
            add_109,
            add_11,
            add_110,
            add_111,
            add_113,
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
            add_57,
            add_59,
            add_6,
            add_61,
            add_62,
            add_63,
            add_65,
            add_66,
            add_67,
            add_68,
            add_69,
            add_7,
            add_71,
            add_72,
            add_73,
            add_75,
            add_76,
            add_78,
            add_8,
            add_80,
            add_81,
            add_82,
            add_84,
            add_85,
            add_86,
            add_87,
            add_88,
            add_9,
            add_90,
            add_91,
            add_92,
            add_94,
            add_95,
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
            concat_0,
            concat_1,
            concat_2,
            conv2d_0,
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
            flatten_1,
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
            full_0,
            full_13,
            full_14,
            full_18,
            full_19,
            full_21,
            full_22,
            full_24,
            full_25,
            full_29,
            full_5,
            full_8,
            full_9,
            full_int_array_16,
            full_int_array_36,
            full_int_array_38,
            full_int_array_39,
            full_int_array_40,
            full_int_array_6,
            full_int_array_7,
            full_int_array_8,
            full_int_array_9,
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
            layer_norm_81,
            layer_norm_82,
            layer_norm_83,
            layer_norm_85,
            layer_norm_86,
            layer_norm_9,
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
            matmul_75,
            matmul_9,
            multiply_0,
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
            multiply_21,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
            reshape_0,
            reshape_1,
            reshape_101,
            reshape_102,
            reshape_104,
            reshape_105,
            reshape_106,
            reshape_108,
            reshape_109,
            reshape_11,
            reshape_114,
            reshape_116,
            reshape_118,
            reshape_119,
            reshape_12,
            reshape_121,
            reshape_122,
            reshape_123,
            reshape_125,
            reshape_126,
            reshape_128,
            reshape_13,
            reshape_130,
            reshape_131,
            reshape_133,
            reshape_134,
            reshape_135,
            reshape_137,
            reshape_138,
            reshape_143,
            reshape_145,
            reshape_147,
            reshape_148,
            reshape_15,
            reshape_150,
            reshape_151,
            reshape_152,
            reshape_154,
            reshape_155,
            reshape_157,
            reshape_158,
            reshape_16,
            reshape_160,
            reshape_162,
            reshape_163,
            reshape_165,
            reshape_166,
            reshape_167,
            reshape_169,
            reshape_170,
            reshape_175,
            reshape_177,
            reshape_179,
            reshape_180,
            reshape_182,
            reshape_183,
            reshape_21,
            reshape_23,
            reshape_25,
            reshape_26,
            reshape_28,
            reshape_29,
            reshape_3,
            reshape_30,
            reshape_32,
            reshape_33,
            reshape_35,
            reshape_36,
            reshape_38,
            reshape_4,
            reshape_40,
            reshape_41,
            reshape_43,
            reshape_44,
            reshape_45,
            reshape_47,
            reshape_48,
            reshape_53,
            reshape_55,
            reshape_57,
            reshape_58,
            reshape_6,
            reshape_60,
            reshape_61,
            reshape_62,
            reshape_64,
            reshape_65,
            reshape_67,
            reshape_68,
            reshape_70,
            reshape_72,
            reshape_73,
            reshape_75,
            reshape_76,
            reshape_77,
            reshape_79,
            reshape_8,
            reshape_80,
            reshape_85,
            reshape_87,
            reshape_89,
            reshape_9,
            reshape_90,
            reshape_92,
            reshape_93,
            reshape_94,
            reshape_96,
            reshape_97,
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
            scale_0,
            scale_1,
            scale_10,
            scale_11,
            scale_2,
            scale_3,
            scale_4,
            scale_5,
            scale_6,
            scale_7,
            scale_8,
            scale_9,
            set_value__17,
            set_value__26,
            set_value__35,
            set_value__44,
            set_value__53,
            set_value__8,
            slice_11,
            slice_14,
            slice_17,
            slice_2,
            slice_20,
            slice_23,
            slice_26,
            slice_29,
            slice_32,
            slice_35,
            slice_5,
            slice_8,
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
            squeeze_0,
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
            transpose_17,
            transpose_18,
            transpose_19,
            transpose_2,
            transpose_20,
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
            transpose_74,
            transpose_75,
            transpose_76,
            transpose_77,
            transpose_78,
            transpose_79,
            transpose_9,
            unsqueeze_0,
            unsqueeze_11,
            unsqueeze_12,
            unsqueeze_15,
            unsqueeze_17,
            unsqueeze_18,
            unsqueeze_21,
            unsqueeze_23,
            unsqueeze_24,
            unsqueeze_27,
            unsqueeze_29,
            unsqueeze_3,
            unsqueeze_30,
            unsqueeze_33,
            unsqueeze_35,
            unsqueeze_36,
            unsqueeze_5,
            unsqueeze_6,
            unsqueeze_9,
        )

        return add_0
