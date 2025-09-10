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
        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_12, parameter_160, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_12, parameter_160

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_159, full_int_array_0)
        del full_int_array_0, parameter_159

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.shape64: (4xi64) <- (-1x96x56x56xf32)
        shape64_0 = paddle._C_ops.shape64(add_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.flatten: (-1x96x3136xf32) <- (-1x96x56x56xf32)
        flatten_0 = paddle._C_ops.flatten(add_1, 2, 3)
        del add_1

        # pd_op.transpose: (-1x3136x96xf32) <- (-1x96x3136xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.layer_norm: (-1x3136x96xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_158, parameter_157, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_157, parameter_158, transpose_0

        # pd_op.shape64: (3xi64) <- (-1x3136x96xf32)
        shape64_1 = paddle._C_ops.shape64(layer_norm_0)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # pd_op.layer_norm: (-1x3136x96xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_0, parameter_156, parameter_155, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_155, parameter_156

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("56"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("96"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [slice_1, full_0, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x3136x96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(layer_norm_3, stack_0)
        del layer_norm_3, stack_0

        # pd_op.shape64: (4xi64) <- (-1x56x56x96xf32)
        shape64_2 = paddle._C_ops.shape64(reshape_1)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_2

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_1 = [slice_2, full_2, full_3, full_2, full_3, full_1]
        del slice_2

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x8x7x8x7x96xf32) <- (-1x56x56x96xf32, 6xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, stack_1)
        del reshape_1, stack_1

        # pd_op.transpose: (-1x8x8x7x7x96xf32) <- (-1x8x7x8x7x96xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_2, [0, 1, 3, 2, 4, 5])
        del reshape_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [-1, 7, 7, 96]

        # pd_op.reshape: (-1x7x7x96xf32) <- (-1x8x8x7x7x96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_1, full_int_array_3)
        del transpose_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [-1, 49, 96]

        # pd_op.reshape: (-1x49x96xf32) <- (-1x7x7x96xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(reshape_3, full_int_array_4)
        del reshape_3

        # pd_op.shape64: (3xi64) <- (-1x49x96xf32)
        shape64_3 = paddle._C_ops.shape64(reshape_4)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # pd_op.matmul: (-1x49x288xf32) <- (-1x49x96xf32, 96x288xf32)
        matmul_0 = paddle._C_ops.matmul(reshape_4, parameter_154, False, False)
        del parameter_154, reshape_4

        # pd_op.add: (-1x49x288xf32) <- (-1x49x288xf32, 288xf32)
        add_2 = paddle._C_ops.add(matmul_0, parameter_153)
        del matmul_0, parameter_153

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("49"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_2 = [slice_3, full_4, full_5, full_5, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (-1x49x3x3x32xf32) <- (-1x49x288xf32, 5xi64)
        reshape_5 = paddle._C_ops.reshape(add_2, stack_2)
        del add_2, stack_2

        # pd_op.transpose: (3x-1x3x49x32xf32) <- (-1x49x3x3x32xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_5, [2, 0, 3, 1, 4])
        del reshape_5

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [3]

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            transpose_2, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_2

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x3x49x32xf32) <- (-1x3x49x32xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_4, full_7, float("0"), True)
        del slice_4

        # pd_op.transpose: (-1x3x32x49xf32) <- (-1x3x49x32xf32)
        transpose_3 = paddle._C_ops.transpose(slice_5, [0, 1, 3, 2])
        del slice_5

        # pd_op.matmul: (-1x3x49x49xf32) <- (-1x3x49x32xf32, -1x3x32x49xf32)
        matmul_1 = paddle._C_ops.matmul(scale_0, transpose_3, False, False)
        del scale_0, transpose_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-1]

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_6 = paddle._C_ops.reshape(data_13, full_int_array_7)
        del data_13

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_0 = paddle._C_ops.index_select(data_0, reshape_6, 0)
        del data_0, reshape_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [49, 49, -1]

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(index_select_0, full_int_array_8)
        del index_select_0

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_7, [2, 0, 1])
        del reshape_7

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(transpose_4, full_int_array_1)
        del transpose_4

        # pd_op.add: (-1x3x49x49xf32) <- (-1x3x49x49xf32, 1x3x49x49xf32)
        add_3 = paddle._C_ops.add(matmul_1, unsqueeze_0)
        del matmul_1, unsqueeze_0

        # pd_op.softmax: (-1x3x49x49xf32) <- (-1x3x49x49xf32)
        softmax_0 = paddle._C_ops.softmax(add_3, -1)
        del add_3

        # pd_op.matmul: (-1x3x49x32xf32) <- (-1x3x49x49xf32, -1x3x49x32xf32)
        matmul_2 = paddle._C_ops.matmul(softmax_0, slice_6, False, False)
        del slice_6, softmax_0

        # pd_op.transpose: (-1x49x3x32xf32) <- (-1x3x49x32xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])
        del matmul_2

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_3 = [slice_3, full_4, full_1]
        del slice_3

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (-1x49x96xf32) <- (-1x49x3x32xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_5, stack_3)
        del stack_3, transpose_5

        # pd_op.matmul: (-1x49x96xf32) <- (-1x49x96xf32, 96x96xf32)
        matmul_3 = paddle._C_ops.matmul(reshape_8, parameter_152, False, False)
        del parameter_152, reshape_8

        # pd_op.add: (-1x49x96xf32) <- (-1x49x96xf32, 96xf32)
        add_4 = paddle._C_ops.add(matmul_3, parameter_151)
        del matmul_3, parameter_151

        # pd_op.reshape: (-1x7x7x96xf32) <- (-1x49x96xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_4, full_int_array_3)
        del add_4

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_9 = [-1, 8, 8, 7, 7, 96]

        # pd_op.reshape: (-1x8x8x7x7x96xf32) <- (-1x7x7x96xf32, 6xi64)
        reshape_10 = paddle._C_ops.reshape(reshape_9, full_int_array_9)
        del reshape_9

        # pd_op.transpose: (-1x8x7x8x7x96xf32) <- (-1x8x8x7x7x96xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_10, [0, 1, 3, 2, 4, 5])
        del reshape_10

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [-1, 56, 56, 96]

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x8x7x8x7x96xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_6, full_int_array_10)
        del transpose_6

        # pd_op.full: (xi64) <- ()
        full_8 = paddle._C_ops.full(
            [], float("3136"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [slice_1, full_8, full_1]
        del slice_1

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (-1x3136x96xf32) <- (-1x56x56x96xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(reshape_11, stack_4)
        del reshape_11, stack_4

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, -1x3136x96xf32)
        add_5 = paddle._C_ops.add(layer_norm_0, reshape_12)
        del layer_norm_0, reshape_12

        # pd_op.layer_norm: (-1x3136x96xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_150, parameter_149, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_149, parameter_150

        # pd_op.matmul: (-1x3136x384xf32) <- (-1x3136x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_148, False, False)
        del layer_norm_6, parameter_148

        # pd_op.add: (-1x3136x384xf32) <- (-1x3136x384xf32, 384xf32)
        add_6 = paddle._C_ops.add(matmul_4, parameter_147)
        del matmul_4, parameter_147

        # pd_op.gelu: (-1x3136x384xf32) <- (-1x3136x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)
        del add_6

        # pd_op.matmul: (-1x3136x96xf32) <- (-1x3136x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_0, parameter_146, False, False)
        del gelu_0, parameter_146

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, 96xf32)
        add_7 = paddle._C_ops.add(matmul_5, parameter_145)
        del matmul_5, parameter_145

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, -1x3136x96xf32)
        add_8 = paddle._C_ops.add(add_5, add_7)
        del add_5, add_7

        # pd_op.shape64: (3xi64) <- (-1x3136x96xf32)
        shape64_4 = paddle._C_ops.shape64(add_8)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_4

        # pd_op.layer_norm: (-1x3136x96xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x96xf32, 96xf32, 96xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_144, parameter_143, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_143, parameter_144

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [slice_7, full_0, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x3136x96xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(layer_norm_9, stack_5)
        del layer_norm_9, stack_5

        # pd_op.shape64: (4xi64) <- (-1x56x56x96xf32)
        shape64_5 = paddle._C_ops.shape64(reshape_13)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [-3, -3]

        # pd_op.roll: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 2xi64)
        roll_0 = paddle._C_ops.roll(reshape_13, full_int_array_11, [1, 2])
        del reshape_13

        # pd_op.shape64: (4xi64) <- (-1x56x56x96xf32)
        shape64_6 = paddle._C_ops.shape64(roll_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_6 = [slice_9, full_2, full_3, full_2, full_3, full_1]
        del full_2, slice_9

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.reshape: (-1x8x7x8x7x96xf32) <- (-1x56x56x96xf32, 6xi64)
        reshape_14 = paddle._C_ops.reshape(roll_0, stack_6)
        del roll_0, stack_6

        # pd_op.transpose: (-1x8x8x7x7x96xf32) <- (-1x8x7x8x7x96xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_14, [0, 1, 3, 2, 4, 5])
        del reshape_14

        # pd_op.reshape: (-1x7x7x96xf32) <- (-1x8x8x7x7x96xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_7, full_int_array_3)
        del transpose_7

        # pd_op.reshape: (-1x49x96xf32) <- (-1x7x7x96xf32, 3xi64)
        reshape_16 = paddle._C_ops.reshape(reshape_15, full_int_array_4)
        del full_int_array_4, reshape_15

        # pd_op.full: (1x56x56x1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1, 56, 56, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [1, 1]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_9,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_9

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [-7, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [-7, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            set_value__1,
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del set_value__1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [-3, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
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
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [-3, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [2147483647, -7]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [2147483647, -3]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [2147483647, 2147483647]

        # pd_op.set_value_: (1x56x56x1xf32) <- (1x56x56x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__7

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_26 = [1, 8, 7, 8, 7, 1]

        # pd_op.reshape: (1x8x7x8x7x1xf32) <- (1x56x56x1xf32, 6xi64)
        reshape_17 = paddle._C_ops.reshape(set_value__8, full_int_array_26)
        del full_int_array_26

        # pd_op.transpose: (1x8x8x7x7x1xf32) <- (1x8x7x8x7x1xf32)
        transpose_8 = paddle._C_ops.transpose(reshape_17, [0, 1, 3, 2, 4, 5])
        del reshape_17

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [-1, 7, 7, 1]

        # pd_op.reshape: (64x7x7x1xf32) <- (1x8x8x7x7x1xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(transpose_8, full_int_array_27)
        del transpose_8

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [-1, 49]

        # pd_op.reshape: (64x49xf32) <- (64x7x7x1xf32, 2xi64)
        reshape_19 = paddle._C_ops.reshape(reshape_18, full_int_array_28)
        del reshape_18

        # pd_op.unsqueeze: (64x1x49xf32) <- (64x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_19, full_int_array_2)

        # pd_op.unsqueeze: (64x49x1xf32) <- (64x49xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(reshape_19, full_int_array_5)
        del reshape_19

        # pd_op.subtract: (64x49x49xf32) <- (64x1x49xf32, 64x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_1, unsqueeze_2)
        del unsqueeze_1, unsqueeze_2

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_10)

        # pd_op.full: (64x49x49xf32) <- ()
        full_11 = paddle._C_ops.full(
            [64, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        where_0 = paddle._C_ops.where(not_equal_0, full_11, subtract_0)
        del full_11, not_equal_0, subtract_0

        # pd_op.equal: (64x49x49xb) <- (64x49x49xf32, xf32)
        equal_0 = paddle._C_ops.equal(where_0, full_10)

        # pd_op.full: (64x49x49xf32) <- ()
        full_12 = paddle._C_ops.full(
            [64, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (64x49x49xf32) <- (64x49x49xb, 64x49x49xf32, 64x49x49xf32)
        where_1 = paddle._C_ops.where(equal_0, full_12, where_0)
        del equal_0, full_12, where_0

        # pd_op.shape64: (3xi64) <- (-1x49x96xf32)
        shape64_7 = paddle._C_ops.shape64(reshape_16)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_7

        # pd_op.matmul: (-1x49x288xf32) <- (-1x49x96xf32, 96x288xf32)
        matmul_6 = paddle._C_ops.matmul(reshape_16, parameter_142, False, False)
        del parameter_142, reshape_16

        # pd_op.add: (-1x49x288xf32) <- (-1x49x288xf32, 288xf32)
        add_9 = paddle._C_ops.add(matmul_6, parameter_141)
        del matmul_6, parameter_141

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_7 = [slice_10, full_4, full_5, full_5, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (-1x49x3x3x32xf32) <- (-1x49x288xf32, 5xi64)
        reshape_20 = paddle._C_ops.reshape(add_9, stack_7)
        del add_9, stack_7

        # pd_op.transpose: (3x-1x3x49x32xf32) <- (-1x49x3x3x32xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_20, [2, 0, 3, 1, 4])
        del reshape_20

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x3x49x32xf32) <- (3x-1x3x49x32xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            transpose_9, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_9

        # pd_op.scale: (-1x3x49x32xf32) <- (-1x3x49x32xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_11, full_7, float("0"), True)
        del slice_11

        # pd_op.transpose: (-1x3x32x49xf32) <- (-1x3x49x32xf32)
        transpose_10 = paddle._C_ops.transpose(slice_12, [0, 1, 3, 2])
        del slice_12

        # pd_op.matmul: (-1x3x49x49xf32) <- (-1x3x49x32xf32, -1x3x32x49xf32)
        matmul_7 = paddle._C_ops.matmul(scale_1, transpose_10, False, False)
        del scale_1, transpose_10

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_21 = paddle._C_ops.reshape(data_14, full_int_array_7)
        del data_14

        # pd_op.index_select: (2401x3xf32) <- (169x3xf32, 2401xi64)
        index_select_1 = paddle._C_ops.index_select(data_1, reshape_21, 0)
        del data_1, reshape_21

        # pd_op.reshape: (49x49x3xf32) <- (2401x3xf32, 3xi64)
        reshape_22 = paddle._C_ops.reshape(index_select_1, full_int_array_8)
        del index_select_1

        # pd_op.transpose: (3x49x49xf32) <- (49x49x3xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_22, [2, 0, 1])
        del reshape_22

        # pd_op.unsqueeze: (1x3x49x49xf32) <- (3x49x49xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(transpose_11, full_int_array_1)
        del transpose_11

        # pd_op.add: (-1x3x49x49xf32) <- (-1x3x49x49xf32, 1x3x49x49xf32)
        add_10 = paddle._C_ops.add(matmul_7, unsqueeze_3)
        del matmul_7, unsqueeze_3

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(slice_10, full_13)
        del full_13

        # pd_op.full: (xi64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("64"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_8 = [floor_divide_0, full_14, full_5, full_4, full_4]
        del floor_divide_0, full_14

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.reshape: (-1x64x3x49x49xf32) <- (-1x3x49x49xf32, 5xi64)
        reshape_23 = paddle._C_ops.reshape(add_10, stack_8)
        del add_10, stack_8

        # pd_op.unsqueeze: (64x1x49x49xf32) <- (64x49x49xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(where_1, full_int_array_2)
        del where_1

        # pd_op.unsqueeze: (1x64x1x49x49xf32) <- (64x1x49x49xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(unsqueeze_4, full_int_array_1)
        del unsqueeze_4

        # pd_op.add: (-1x64x3x49x49xf32) <- (-1x64x3x49x49xf32, 1x64x1x49x49xf32)
        add_11 = paddle._C_ops.add(reshape_23, unsqueeze_5)
        del reshape_23, unsqueeze_5

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_9 = [slice_10, full_5, full_4, full_4]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_9, 0)
        del combine_9

        # pd_op.reshape: (-1x3x49x49xf32) <- (-1x64x3x49x49xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_11, stack_9)
        del add_11, stack_9

        # pd_op.softmax: (-1x3x49x49xf32) <- (-1x3x49x49xf32)
        softmax_1 = paddle._C_ops.softmax(reshape_24, -1)
        del reshape_24

        # pd_op.matmul: (-1x3x49x32xf32) <- (-1x3x49x49xf32, -1x3x49x32xf32)
        matmul_8 = paddle._C_ops.matmul(softmax_1, slice_13, False, False)
        del slice_13, softmax_1

        # pd_op.transpose: (-1x49x3x32xf32) <- (-1x3x49x32xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])
        del matmul_8

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_10 = [slice_10, full_4, full_1]
        del slice_10

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.reshape: (-1x49x96xf32) <- (-1x49x3x32xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_12, stack_10)
        del stack_10, transpose_12

        # pd_op.matmul: (-1x49x96xf32) <- (-1x49x96xf32, 96x96xf32)
        matmul_9 = paddle._C_ops.matmul(reshape_25, parameter_140, False, False)
        del parameter_140, reshape_25

        # pd_op.add: (-1x49x96xf32) <- (-1x49x96xf32, 96xf32)
        add_12 = paddle._C_ops.add(matmul_9, parameter_139)
        del matmul_9, parameter_139

        # pd_op.reshape: (-1x7x7x96xf32) <- (-1x49x96xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_12, full_int_array_3)
        del add_12, full_int_array_3

        # pd_op.reshape: (-1x8x8x7x7x96xf32) <- (-1x7x7x96xf32, 6xi64)
        reshape_27 = paddle._C_ops.reshape(reshape_26, full_int_array_9)
        del full_int_array_9, reshape_26

        # pd_op.transpose: (-1x8x7x8x7x96xf32) <- (-1x8x8x7x7x96xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_27, [0, 1, 3, 2, 4, 5])
        del reshape_27

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x8x7x8x7x96xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(transpose_13, full_int_array_10)
        del full_int_array_10, transpose_13

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_29 = [3, 3]

        # pd_op.roll: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 2xi64)
        roll_1 = paddle._C_ops.roll(reshape_28, full_int_array_29, [1, 2])
        del reshape_28

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [slice_7, full_8, full_1]
        del full_8, slice_7

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_11, 0)
        del combine_11

        # pd_op.reshape: (-1x3136x96xf32) <- (-1x56x56x96xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(roll_1, stack_11)
        del roll_1, stack_11

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, -1x3136x96xf32)
        add_13 = paddle._C_ops.add(add_8, reshape_29)
        del add_8, reshape_29

        # pd_op.layer_norm: (-1x3136x96xf32, -1x3136xf32, -1x3136xf32) <- (-1x3136x96xf32, 96xf32, 96xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_13, parameter_138, parameter_137, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_137, parameter_138

        # pd_op.matmul: (-1x3136x384xf32) <- (-1x3136x96xf32, 96x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_12, parameter_136, False, False)
        del layer_norm_12, parameter_136

        # pd_op.add: (-1x3136x384xf32) <- (-1x3136x384xf32, 384xf32)
        add_14 = paddle._C_ops.add(matmul_10, parameter_135)
        del matmul_10, parameter_135

        # pd_op.gelu: (-1x3136x384xf32) <- (-1x3136x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_14, False)
        del add_14

        # pd_op.matmul: (-1x3136x96xf32) <- (-1x3136x384xf32, 384x96xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_1, parameter_134, False, False)
        del gelu_1, parameter_134

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, 96xf32)
        add_15 = paddle._C_ops.add(matmul_11, parameter_133)
        del matmul_11, parameter_133

        # pd_op.add: (-1x3136x96xf32) <- (-1x3136x96xf32, -1x3136x96xf32)
        add_16 = paddle._C_ops.add(add_13, add_15)
        del add_13, add_15

        # pd_op.shape64: (3xi64) <- (-1x3136x96xf32)
        shape64_8 = paddle._C_ops.shape64(add_16)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_14, full_0, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x3136x96xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_16, stack_12)
        del add_16, stack_12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_30 = [2, 2]

        # pd_op.strided_slice: (-1x28x28x96xf32) <- (-1x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_12, full_int_array_25, full_int_array_30
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [1, 0]

        # pd_op.strided_slice: (-1x28x28x96xf32) <- (-1x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_31, full_int_array_25, full_int_array_30
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [0, 1]

        # pd_op.strided_slice: (-1x28x28x96xf32) <- (-1x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_32, full_int_array_25, full_int_array_30
        )

        # pd_op.strided_slice: (-1x28x28x96xf32) <- (-1x56x56x96xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            reshape_30, [1, 2], full_int_array_14, full_int_array_25, full_int_array_30
        )

        # pd_op.shape64: (4xi64) <- (-1x56x56x96xf32)
        shape64_9 = paddle._C_ops.shape64(reshape_30)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_13 = [slice_15, full_0, full_0, full_1]
        del full_0, full_1, slice_15

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.reshape: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(reshape_30, stack_13)
        del reshape_30, stack_13

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32]) <- (-1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32)
        combine_14 = [
            strided_slice_0,
            strided_slice_1,
            strided_slice_2,
            strided_slice_3,
        ]
        del strided_slice_0, strided_slice_1, strided_slice_2, strided_slice_3

        # pd_op.concat: (-1x28x28x384xf32) <- ([-1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32, -1x28x28x96xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_14, full_15)
        del combine_14

        # pd_op.full: (xi64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("384"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_15 = [slice_14, full_16, full_17]
        del slice_14

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_15, 0)
        del combine_15

        # pd_op.reshape: (-1x-1x384xf32) <- (-1x28x28x384xf32, 3xi64)
        reshape_32 = paddle._C_ops.reshape(concat_0, stack_14)
        del concat_0, stack_14

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_32, parameter_132, parameter_131, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_131, parameter_132, reshape_32

        # pd_op.matmul: (-1x-1x192xf32) <- (-1x-1x384xf32, 384x192xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_130, False, False)
        del layer_norm_15, parameter_130

        # pd_op.shape64: (3xi64) <- (-1x-1x192xf32)
        shape64_10 = paddle._C_ops.shape64(matmul_12)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_10

        # pd_op.shape64: (3xi64) <- (-1x-1x192xf32)
        shape64_11 = paddle._C_ops.shape64(matmul_12)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_2, full_int_array_5, [1], [0]
        )
        del shape64_11

        # pd_op.layer_norm: (-1x-1x192xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x192xf32, 192xf32, 192xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_12, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_128, parameter_129

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("28"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("192"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_16 = [slice_16, full_18, full_18, full_19]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x-1x192xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(layer_norm_18, stack_15)
        del layer_norm_18, stack_15

        # pd_op.shape64: (4xi64) <- (-1x28x28x192xf32)
        shape64_12 = paddle._C_ops.shape64(reshape_33)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_12

        # pd_op.full: (xi64) <- ()
        full_20 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_17 = [slice_18, full_20, full_3, full_20, full_3, full_19]
        del slice_18

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_17, 0)
        del combine_17

        # pd_op.reshape: (-1x4x7x4x7x192xf32) <- (-1x28x28x192xf32, 6xi64)
        reshape_34 = paddle._C_ops.reshape(reshape_33, stack_16)
        del reshape_33, stack_16

        # pd_op.transpose: (-1x4x4x7x7x192xf32) <- (-1x4x7x4x7x192xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_34, [0, 1, 3, 2, 4, 5])
        del reshape_34

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [-1, 7, 7, 192]

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x4x4x7x7x192xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_14, full_int_array_33)
        del transpose_14

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_34 = [-1, 49, 192]

        # pd_op.reshape: (-1x49x192xf32) <- (-1x7x7x192xf32, 3xi64)
        reshape_36 = paddle._C_ops.reshape(reshape_35, full_int_array_34)
        del reshape_35

        # pd_op.shape64: (3xi64) <- (-1x49x192xf32)
        shape64_13 = paddle._C_ops.shape64(reshape_36)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_13

        # pd_op.matmul: (-1x49x576xf32) <- (-1x49x192xf32, 192x576xf32)
        matmul_13 = paddle._C_ops.matmul(reshape_36, parameter_127, False, False)
        del parameter_127, reshape_36

        # pd_op.add: (-1x49x576xf32) <- (-1x49x576xf32, 576xf32)
        add_17 = paddle._C_ops.add(matmul_13, parameter_126)
        del matmul_13, parameter_126

        # pd_op.full: (xi64) <- ()
        full_21 = paddle._C_ops.full(
            [], float("6"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_18 = [slice_19, full_4, full_5, full_21, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.reshape: (-1x49x3x6x32xf32) <- (-1x49x576xf32, 5xi64)
        reshape_37 = paddle._C_ops.reshape(add_17, stack_17)
        del add_17, stack_17

        # pd_op.transpose: (3x-1x6x49x32xf32) <- (-1x49x3x6x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_37, [2, 0, 3, 1, 4])
        del reshape_37

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            transpose_15, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_15

        # pd_op.scale: (-1x6x49x32xf32) <- (-1x6x49x32xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_20, full_7, float("0"), True)
        del slice_20

        # pd_op.transpose: (-1x6x32x49xf32) <- (-1x6x49x32xf32)
        transpose_16 = paddle._C_ops.transpose(slice_21, [0, 1, 3, 2])
        del slice_21

        # pd_op.matmul: (-1x6x49x49xf32) <- (-1x6x49x32xf32, -1x6x32x49xf32)
        matmul_14 = paddle._C_ops.matmul(scale_2, transpose_16, False, False)
        del scale_2, transpose_16

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_38 = paddle._C_ops.reshape(data_15, full_int_array_7)
        del data_15

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_2 = paddle._C_ops.index_select(data_2, reshape_38, 0)
        del data_2, reshape_38

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(index_select_2, full_int_array_8)
        del index_select_2

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_17 = paddle._C_ops.transpose(reshape_39, [2, 0, 1])
        del reshape_39

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(transpose_17, full_int_array_1)
        del transpose_17

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x49x49xf32, 1x6x49x49xf32)
        add_18 = paddle._C_ops.add(matmul_14, unsqueeze_6)
        del matmul_14, unsqueeze_6

        # pd_op.softmax: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax_2 = paddle._C_ops.softmax(add_18, -1)
        del add_18

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x49x32xf32)
        matmul_15 = paddle._C_ops.matmul(softmax_2, slice_22, False, False)
        del slice_22, softmax_2

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_18 = paddle._C_ops.transpose(matmul_15, [0, 2, 1, 3])
        del matmul_15

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_19 = [slice_19, full_4, full_19]
        del slice_19

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.reshape: (-1x49x192xf32) <- (-1x49x6x32xf32, 3xi64)
        reshape_40 = paddle._C_ops.reshape(transpose_18, stack_18)
        del stack_18, transpose_18

        # pd_op.matmul: (-1x49x192xf32) <- (-1x49x192xf32, 192x192xf32)
        matmul_16 = paddle._C_ops.matmul(reshape_40, parameter_125, False, False)
        del parameter_125, reshape_40

        # pd_op.add: (-1x49x192xf32) <- (-1x49x192xf32, 192xf32)
        add_19 = paddle._C_ops.add(matmul_16, parameter_124)
        del matmul_16, parameter_124

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x49x192xf32, 4xi64)
        reshape_41 = paddle._C_ops.reshape(add_19, full_int_array_33)
        del add_19

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_35 = [-1, 4, 4, 7, 7, 192]

        # pd_op.reshape: (-1x4x4x7x7x192xf32) <- (-1x7x7x192xf32, 6xi64)
        reshape_42 = paddle._C_ops.reshape(reshape_41, full_int_array_35)
        del reshape_41

        # pd_op.transpose: (-1x4x7x4x7x192xf32) <- (-1x4x4x7x7x192xf32)
        transpose_19 = paddle._C_ops.transpose(reshape_42, [0, 1, 3, 2, 4, 5])
        del reshape_42

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [-1, 28, 28, 192]

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x4x7x4x7x192xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(transpose_19, full_int_array_36)
        del transpose_19

        # pd_op.full: (xi64) <- ()
        full_22 = paddle._C_ops.full(
            [], float("784"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_20 = [slice_16, full_22, full_19]
        del slice_16

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.reshape: (-1x784x192xf32) <- (-1x28x28x192xf32, 3xi64)
        reshape_44 = paddle._C_ops.reshape(reshape_43, stack_19)
        del reshape_43, stack_19

        # pd_op.add: (-1x784x192xf32) <- (-1x-1x192xf32, -1x784x192xf32)
        add_20 = paddle._C_ops.add(matmul_12, reshape_44)
        del matmul_12, reshape_44

        # pd_op.layer_norm: (-1x784x192xf32, -1x784xf32, -1x784xf32) <- (-1x784x192xf32, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_123, parameter_122, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_122, parameter_123

        # pd_op.matmul: (-1x784x768xf32) <- (-1x784x192xf32, 192x768xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_21, parameter_121, False, False)
        del layer_norm_21, parameter_121

        # pd_op.add: (-1x784x768xf32) <- (-1x784x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_17, parameter_120)
        del matmul_17, parameter_120

        # pd_op.gelu: (-1x784x768xf32) <- (-1x784x768xf32)
        gelu_2 = paddle._C_ops.gelu(add_21, False)
        del add_21

        # pd_op.matmul: (-1x784x192xf32) <- (-1x784x768xf32, 768x192xf32)
        matmul_18 = paddle._C_ops.matmul(gelu_2, parameter_119, False, False)
        del gelu_2, parameter_119

        # pd_op.add: (-1x784x192xf32) <- (-1x784x192xf32, 192xf32)
        add_22 = paddle._C_ops.add(matmul_18, parameter_118)
        del matmul_18, parameter_118

        # pd_op.add: (-1x784x192xf32) <- (-1x784x192xf32, -1x784x192xf32)
        add_23 = paddle._C_ops.add(add_20, add_22)
        del add_20, add_22

        # pd_op.shape64: (3xi64) <- (-1x784x192xf32)
        shape64_14 = paddle._C_ops.shape64(add_23)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_14

        # pd_op.layer_norm: (-1x784x192xf32, -1x784xf32, -1x784xf32) <- (-1x784x192xf32, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_117, parameter_116, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_21 = [slice_23, full_18, full_18, full_19]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_21, 0)
        del combine_21

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x784x192xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(layer_norm_24, stack_20)
        del layer_norm_24, stack_20

        # pd_op.shape64: (4xi64) <- (-1x28x28x192xf32)
        shape64_15 = paddle._C_ops.shape64(reshape_45)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_15

        # pd_op.roll: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 2xi64)
        roll_2 = paddle._C_ops.roll(reshape_45, full_int_array_11, [1, 2])
        del reshape_45

        # pd_op.shape64: (4xi64) <- (-1x28x28x192xf32)
        shape64_16 = paddle._C_ops.shape64(roll_2)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_16

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_22 = [slice_25, full_20, full_3, full_20, full_3, full_19]
        del slice_25

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.reshape: (-1x4x7x4x7x192xf32) <- (-1x28x28x192xf32, 6xi64)
        reshape_46 = paddle._C_ops.reshape(roll_2, stack_21)
        del roll_2, stack_21

        # pd_op.transpose: (-1x4x4x7x7x192xf32) <- (-1x4x7x4x7x192xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_46, [0, 1, 3, 2, 4, 5])
        del reshape_46

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x4x4x7x7x192xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_20, full_int_array_33)
        del transpose_20

        # pd_op.reshape: (-1x49x192xf32) <- (-1x7x7x192xf32, 3xi64)
        reshape_48 = paddle._C_ops.reshape(reshape_47, full_int_array_34)
        del full_int_array_34, reshape_47

        # pd_op.full: (1x28x28x1xf32) <- ()
        full_23 = paddle._C_ops.full(
            [1, 28, 28, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__9 = paddle._C_ops.set_value_(
            full_23,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_23

        # pd_op.set_value_: (1x28x28x1xf32) <- (1x28x28x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__10 = paddle._C_ops.set_value_(
            set_value__9,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
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
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
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
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
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
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
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
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
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
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
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
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
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
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__16

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_37 = [1, 4, 7, 4, 7, 1]

        # pd_op.reshape: (1x4x7x4x7x1xf32) <- (1x28x28x1xf32, 6xi64)
        reshape_49 = paddle._C_ops.reshape(set_value__17, full_int_array_37)
        del full_int_array_37

        # pd_op.transpose: (1x4x4x7x7x1xf32) <- (1x4x7x4x7x1xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_49, [0, 1, 3, 2, 4, 5])
        del reshape_49

        # pd_op.reshape: (16x7x7x1xf32) <- (1x4x4x7x7x1xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_21, full_int_array_27)
        del transpose_21

        # pd_op.reshape: (16x49xf32) <- (16x7x7x1xf32, 2xi64)
        reshape_51 = paddle._C_ops.reshape(reshape_50, full_int_array_28)
        del reshape_50

        # pd_op.unsqueeze: (16x1x49xf32) <- (16x49xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(reshape_51, full_int_array_2)

        # pd_op.unsqueeze: (16x49x1xf32) <- (16x49xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(reshape_51, full_int_array_5)
        del reshape_51

        # pd_op.subtract: (16x49x49xf32) <- (16x1x49xf32, 16x49x1xf32)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_7, unsqueeze_8)
        del unsqueeze_7, unsqueeze_8

        # pd_op.not_equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        not_equal_1 = paddle._C_ops.not_equal(subtract_1, full_10)

        # pd_op.full: (16x49x49xf32) <- ()
        full_24 = paddle._C_ops.full(
            [16, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        where_2 = paddle._C_ops.where(not_equal_1, full_24, subtract_1)
        del full_24, not_equal_1, subtract_1

        # pd_op.equal: (16x49x49xb) <- (16x49x49xf32, xf32)
        equal_1 = paddle._C_ops.equal(where_2, full_10)

        # pd_op.full: (16x49x49xf32) <- ()
        full_25 = paddle._C_ops.full(
            [16, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (16x49x49xf32) <- (16x49x49xb, 16x49x49xf32, 16x49x49xf32)
        where_3 = paddle._C_ops.where(equal_1, full_25, where_2)
        del equal_1, full_25, where_2

        # pd_op.shape64: (3xi64) <- (-1x49x192xf32)
        shape64_17 = paddle._C_ops.shape64(reshape_48)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_17

        # pd_op.matmul: (-1x49x576xf32) <- (-1x49x192xf32, 192x576xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_48, parameter_115, False, False)
        del parameter_115, reshape_48

        # pd_op.add: (-1x49x576xf32) <- (-1x49x576xf32, 576xf32)
        add_24 = paddle._C_ops.add(matmul_19, parameter_114)
        del matmul_19, parameter_114

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_23 = [slice_26, full_4, full_5, full_21, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_23, 0)
        del combine_23

        # pd_op.reshape: (-1x49x3x6x32xf32) <- (-1x49x576xf32, 5xi64)
        reshape_52 = paddle._C_ops.reshape(add_24, stack_22)
        del add_24, stack_22

        # pd_op.transpose: (3x-1x6x49x32xf32) <- (-1x49x3x6x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_52, [2, 0, 3, 1, 4])
        del reshape_52

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x6x49x32xf32) <- (3x-1x6x49x32xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(
            transpose_22, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_22

        # pd_op.scale: (-1x6x49x32xf32) <- (-1x6x49x32xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(slice_27, full_7, float("0"), True)
        del slice_27

        # pd_op.transpose: (-1x6x32x49xf32) <- (-1x6x49x32xf32)
        transpose_23 = paddle._C_ops.transpose(slice_28, [0, 1, 3, 2])
        del slice_28

        # pd_op.matmul: (-1x6x49x49xf32) <- (-1x6x49x32xf32, -1x6x32x49xf32)
        matmul_20 = paddle._C_ops.matmul(scale_3, transpose_23, False, False)
        del scale_3, transpose_23

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_53 = paddle._C_ops.reshape(data_16, full_int_array_7)
        del data_16

        # pd_op.index_select: (2401x6xf32) <- (169x6xf32, 2401xi64)
        index_select_3 = paddle._C_ops.index_select(data_3, reshape_53, 0)
        del data_3, reshape_53

        # pd_op.reshape: (49x49x6xf32) <- (2401x6xf32, 3xi64)
        reshape_54 = paddle._C_ops.reshape(index_select_3, full_int_array_8)
        del index_select_3

        # pd_op.transpose: (6x49x49xf32) <- (49x49x6xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_54, [2, 0, 1])
        del reshape_54

        # pd_op.unsqueeze: (1x6x49x49xf32) <- (6x49x49xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(transpose_24, full_int_array_1)
        del transpose_24

        # pd_op.add: (-1x6x49x49xf32) <- (-1x6x49x49xf32, 1x6x49x49xf32)
        add_25 = paddle._C_ops.add(matmul_20, unsqueeze_9)
        del matmul_20, unsqueeze_9

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_1 = paddle._C_ops.floor_divide(slice_26, full_26)
        del full_26

        # pd_op.full: (xi64) <- ()
        full_27 = paddle._C_ops.full(
            [], float("16"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_24 = [floor_divide_1, full_27, full_21, full_4, full_4]
        del floor_divide_1, full_27

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.reshape: (-1x16x6x49x49xf32) <- (-1x6x49x49xf32, 5xi64)
        reshape_55 = paddle._C_ops.reshape(add_25, stack_23)
        del add_25, stack_23

        # pd_op.unsqueeze: (16x1x49x49xf32) <- (16x49x49xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(where_3, full_int_array_2)
        del where_3

        # pd_op.unsqueeze: (1x16x1x49x49xf32) <- (16x1x49x49xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(unsqueeze_10, full_int_array_1)
        del unsqueeze_10

        # pd_op.add: (-1x16x6x49x49xf32) <- (-1x16x6x49x49xf32, 1x16x1x49x49xf32)
        add_26 = paddle._C_ops.add(reshape_55, unsqueeze_11)
        del reshape_55, unsqueeze_11

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_25 = [slice_26, full_21, full_4, full_4]
        del full_21

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.reshape: (-1x6x49x49xf32) <- (-1x16x6x49x49xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(add_26, stack_24)
        del add_26, stack_24

        # pd_op.softmax: (-1x6x49x49xf32) <- (-1x6x49x49xf32)
        softmax_3 = paddle._C_ops.softmax(reshape_56, -1)
        del reshape_56

        # pd_op.matmul: (-1x6x49x32xf32) <- (-1x6x49x49xf32, -1x6x49x32xf32)
        matmul_21 = paddle._C_ops.matmul(softmax_3, slice_29, False, False)
        del slice_29, softmax_3

        # pd_op.transpose: (-1x49x6x32xf32) <- (-1x6x49x32xf32)
        transpose_25 = paddle._C_ops.transpose(matmul_21, [0, 2, 1, 3])
        del matmul_21

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_26 = [slice_26, full_4, full_19]
        del slice_26

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.reshape: (-1x49x192xf32) <- (-1x49x6x32xf32, 3xi64)
        reshape_57 = paddle._C_ops.reshape(transpose_25, stack_25)
        del stack_25, transpose_25

        # pd_op.matmul: (-1x49x192xf32) <- (-1x49x192xf32, 192x192xf32)
        matmul_22 = paddle._C_ops.matmul(reshape_57, parameter_113, False, False)
        del parameter_113, reshape_57

        # pd_op.add: (-1x49x192xf32) <- (-1x49x192xf32, 192xf32)
        add_27 = paddle._C_ops.add(matmul_22, parameter_112)
        del matmul_22, parameter_112

        # pd_op.reshape: (-1x7x7x192xf32) <- (-1x49x192xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_27, full_int_array_33)
        del add_27, full_int_array_33

        # pd_op.reshape: (-1x4x4x7x7x192xf32) <- (-1x7x7x192xf32, 6xi64)
        reshape_59 = paddle._C_ops.reshape(reshape_58, full_int_array_35)
        del full_int_array_35, reshape_58

        # pd_op.transpose: (-1x4x7x4x7x192xf32) <- (-1x4x4x7x7x192xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_59, [0, 1, 3, 2, 4, 5])
        del reshape_59

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x4x7x4x7x192xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(transpose_26, full_int_array_36)
        del full_int_array_36, transpose_26

        # pd_op.roll: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 2xi64)
        roll_3 = paddle._C_ops.roll(reshape_60, full_int_array_29, [1, 2])
        del reshape_60

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_27 = [slice_23, full_22, full_19]
        del full_22, slice_23

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_27, 0)
        del combine_27

        # pd_op.reshape: (-1x784x192xf32) <- (-1x28x28x192xf32, 3xi64)
        reshape_61 = paddle._C_ops.reshape(roll_3, stack_26)
        del roll_3, stack_26

        # pd_op.add: (-1x784x192xf32) <- (-1x784x192xf32, -1x784x192xf32)
        add_28 = paddle._C_ops.add(add_23, reshape_61)
        del add_23, reshape_61

        # pd_op.layer_norm: (-1x784x192xf32, -1x784xf32, -1x784xf32) <- (-1x784x192xf32, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_28, parameter_111, parameter_110, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_110, parameter_111

        # pd_op.matmul: (-1x784x768xf32) <- (-1x784x192xf32, 192x768xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_27, parameter_109, False, False)
        del layer_norm_27, parameter_109

        # pd_op.add: (-1x784x768xf32) <- (-1x784x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_23, parameter_108)
        del matmul_23, parameter_108

        # pd_op.gelu: (-1x784x768xf32) <- (-1x784x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_29, False)
        del add_29

        # pd_op.matmul: (-1x784x192xf32) <- (-1x784x768xf32, 768x192xf32)
        matmul_24 = paddle._C_ops.matmul(gelu_3, parameter_107, False, False)
        del gelu_3, parameter_107

        # pd_op.add: (-1x784x192xf32) <- (-1x784x192xf32, 192xf32)
        add_30 = paddle._C_ops.add(matmul_24, parameter_106)
        del matmul_24, parameter_106

        # pd_op.add: (-1x784x192xf32) <- (-1x784x192xf32, -1x784x192xf32)
        add_31 = paddle._C_ops.add(add_28, add_30)
        del add_28, add_30

        # pd_op.shape64: (3xi64) <- (-1x784x192xf32)
        shape64_18 = paddle._C_ops.shape64(add_31)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_18

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_28 = [slice_30, full_18, full_18, full_19]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_27 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x784x192xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(add_31, stack_27)
        del add_31, stack_27

        # pd_op.strided_slice: (-1x14x14x192xf32) <- (-1x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_4 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_12, full_int_array_25, full_int_array_30
        )

        # pd_op.strided_slice: (-1x14x14x192xf32) <- (-1x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_5 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_31, full_int_array_25, full_int_array_30
        )

        # pd_op.strided_slice: (-1x14x14x192xf32) <- (-1x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_6 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_32, full_int_array_25, full_int_array_30
        )

        # pd_op.strided_slice: (-1x14x14x192xf32) <- (-1x28x28x192xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_7 = paddle._C_ops.strided_slice(
            reshape_62, [1, 2], full_int_array_14, full_int_array_25, full_int_array_30
        )

        # pd_op.shape64: (4xi64) <- (-1x28x28x192xf32)
        shape64_19 = paddle._C_ops.shape64(reshape_62)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_19

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_29 = [slice_31, full_18, full_18, full_19]
        del full_18, full_19, slice_31

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_28 = paddle._C_ops.stack(combine_29, 0)
        del combine_29

        # pd_op.reshape: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(reshape_62, stack_28)
        del reshape_62, stack_28

        # builtin.combine: ([-1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32]) <- (-1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32)
        combine_30 = [
            strided_slice_4,
            strided_slice_5,
            strided_slice_6,
            strided_slice_7,
        ]
        del strided_slice_4, strided_slice_5, strided_slice_6, strided_slice_7

        # pd_op.concat: (-1x14x14x768xf32) <- ([-1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32, -1x14x14x192xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_30, full_15)
        del combine_30

        # pd_op.full: (xi64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("768"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_31 = [slice_30, full_16, full_28]
        del slice_30

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_29 = paddle._C_ops.stack(combine_31, 0)
        del combine_31

        # pd_op.reshape: (-1x-1x768xf32) <- (-1x14x14x768xf32, 3xi64)
        reshape_64 = paddle._C_ops.reshape(concat_1, stack_29)
        del concat_1, stack_29

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_64, parameter_105, parameter_104, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_104, parameter_105, reshape_64

        # pd_op.matmul: (-1x-1x384xf32) <- (-1x-1x768xf32, 768x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_30, parameter_103, False, False)
        del layer_norm_30, parameter_103

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_20 = paddle._C_ops.shape64(matmul_25)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_20

        # pd_op.shape64: (3xi64) <- (-1x-1x384xf32)
        shape64_21 = paddle._C_ops.shape64(matmul_25)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_2, full_int_array_5, [1], [0]
        )
        del shape64_21

        # pd_op.layer_norm: (-1x-1x384xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_25, parameter_102, parameter_101, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_101, parameter_102

        # pd_op.full: (xi64) <- ()
        full_29 = paddle._C_ops.full(
            [], float("14"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_32 = [slice_32, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_30 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x-1x384xf32, 4xi64)
        reshape_65 = paddle._C_ops.reshape(layer_norm_33, stack_30)
        del layer_norm_33, stack_30

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_22 = paddle._C_ops.shape64(reshape_65)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_22

        # pd_op.full: (xi64) <- ()
        full_30 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_33 = [slice_34, full_30, full_3, full_30, full_3, full_17]
        del slice_34

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_31 = paddle._C_ops.stack(combine_33, 0)
        del combine_33

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_66 = paddle._C_ops.reshape(reshape_65, stack_31)
        del reshape_65, stack_31

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_27 = paddle._C_ops.transpose(reshape_66, [0, 1, 3, 2, 4, 5])
        del reshape_66

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [-1, 7, 7, 384]

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(transpose_27, full_int_array_38)
        del transpose_27

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_39 = [-1, 49, 384]

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_68 = paddle._C_ops.reshape(reshape_67, full_int_array_39)
        del reshape_67

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_23 = paddle._C_ops.shape64(reshape_68)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_23

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_26 = paddle._C_ops.matmul(reshape_68, parameter_100, False, False)
        del parameter_100, reshape_68

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_32 = paddle._C_ops.add(matmul_26, parameter_99)
        del matmul_26, parameter_99

        # pd_op.full: (xi64) <- ()
        full_31 = paddle._C_ops.full(
            [], float("12"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_34 = [slice_35, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_32 = paddle._C_ops.stack(combine_34, 0)
        del combine_34

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_69 = paddle._C_ops.reshape(add_32, stack_32)
        del add_32, stack_32

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_69, [2, 0, 3, 1, 4])
        del reshape_69

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(
            transpose_28, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_28

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(slice_36, full_7, float("0"), True)
        del slice_36

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_29 = paddle._C_ops.transpose(slice_37, [0, 1, 3, 2])
        del slice_37

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_27 = paddle._C_ops.matmul(scale_4, transpose_29, False, False)
        del scale_4, transpose_29

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_70 = paddle._C_ops.reshape(data_17, full_int_array_7)
        del data_17

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_4 = paddle._C_ops.index_select(data_4, reshape_70, 0)
        del data_4, reshape_70

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(index_select_4, full_int_array_8)
        del index_select_4

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_71, [2, 0, 1])
        del reshape_71

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(transpose_30, full_int_array_1)
        del transpose_30

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_33 = paddle._C_ops.add(matmul_27, unsqueeze_12)
        del matmul_27, unsqueeze_12

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_4 = paddle._C_ops.softmax(add_33, -1)
        del add_33

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_28 = paddle._C_ops.matmul(softmax_4, slice_38, False, False)
        del slice_38, softmax_4

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])
        del matmul_28

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_35 = [slice_35, full_4, full_17]
        del slice_35

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_33 = paddle._C_ops.stack(combine_35, 0)
        del combine_35

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_72 = paddle._C_ops.reshape(transpose_31, stack_33)
        del stack_33, transpose_31

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_29 = paddle._C_ops.matmul(reshape_72, parameter_98, False, False)
        del parameter_98, reshape_72

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_34 = paddle._C_ops.add(matmul_29, parameter_97)
        del matmul_29, parameter_97

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_73 = paddle._C_ops.reshape(add_34, full_int_array_38)
        del add_34

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_40 = [-1, 2, 2, 7, 7, 384]

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_74 = paddle._C_ops.reshape(reshape_73, full_int_array_40)
        del reshape_73

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_74, [0, 1, 3, 2, 4, 5])
        del reshape_74

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [-1, 14, 14, 384]

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_75 = paddle._C_ops.reshape(transpose_32, full_int_array_41)
        del transpose_32

        # pd_op.full: (xi64) <- ()
        full_32 = paddle._C_ops.full(
            [], float("196"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_36 = [slice_32, full_32, full_17]
        del slice_32

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_34 = paddle._C_ops.stack(combine_36, 0)
        del combine_36

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_76 = paddle._C_ops.reshape(reshape_75, stack_34)
        del reshape_75, stack_34

        # pd_op.add: (-1x196x384xf32) <- (-1x-1x384xf32, -1x196x384xf32)
        add_35 = paddle._C_ops.add(matmul_25, reshape_76)
        del matmul_25, reshape_76

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_96, parameter_95, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_95, parameter_96

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_36, parameter_94, False, False)
        del layer_norm_36, parameter_94

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_36 = paddle._C_ops.add(matmul_30, parameter_93)
        del matmul_30, parameter_93

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_4 = paddle._C_ops.gelu(add_36, False)
        del add_36

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_4, parameter_92, False, False)
        del gelu_4, parameter_92

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_31, parameter_91)
        del matmul_31, parameter_91

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_38 = paddle._C_ops.add(add_35, add_37)
        del add_35, add_37

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_24 = paddle._C_ops.shape64(add_38)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(
            shape64_24, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_24

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_38, parameter_90, parameter_89, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_89, parameter_90

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_37 = [slice_39, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_35 = paddle._C_ops.stack(combine_37, 0)
        del combine_37

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_77 = paddle._C_ops.reshape(layer_norm_39, stack_35)
        del layer_norm_39, stack_35

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_25 = paddle._C_ops.shape64(reshape_77)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(
            shape64_25, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_25

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_4 = paddle._C_ops.roll(reshape_77, full_int_array_11, [1, 2])
        del reshape_77

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_26 = paddle._C_ops.shape64(roll_4)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(
            shape64_26, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_26

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_38 = [slice_41, full_30, full_3, full_30, full_3, full_17]
        del slice_41

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_36 = paddle._C_ops.stack(combine_38, 0)
        del combine_38

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_78 = paddle._C_ops.reshape(roll_4, stack_36)
        del roll_4, stack_36

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_33 = paddle._C_ops.transpose(reshape_78, [0, 1, 3, 2, 4, 5])
        del reshape_78

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_79 = paddle._C_ops.reshape(transpose_33, full_int_array_38)
        del transpose_33

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_80 = paddle._C_ops.reshape(reshape_79, full_int_array_39)
        del reshape_79

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_33 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__18 = paddle._C_ops.set_value_(
            full_33,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_33

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__19 = paddle._C_ops.set_value_(
            set_value__18,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
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
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
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
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
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
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
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
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
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
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
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
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
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
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__25

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_42 = [1, 2, 7, 2, 7, 1]

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_81 = paddle._C_ops.reshape(set_value__26, full_int_array_42)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_81, [0, 1, 3, 2, 4, 5])
        del reshape_81

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_82 = paddle._C_ops.reshape(transpose_34, full_int_array_27)
        del transpose_34

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_83 = paddle._C_ops.reshape(reshape_82, full_int_array_28)
        del reshape_82

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(reshape_83, full_int_array_2)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(reshape_83, full_int_array_5)
        del reshape_83

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_2 = paddle._C_ops.subtract(unsqueeze_13, unsqueeze_14)
        del unsqueeze_13, unsqueeze_14

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_2 = paddle._C_ops.not_equal(subtract_2, full_10)

        # pd_op.full: (4x49x49xf32) <- ()
        full_34 = paddle._C_ops.full(
            [4, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_4 = paddle._C_ops.where(not_equal_2, full_34, subtract_2)
        del not_equal_2, subtract_2

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_2 = paddle._C_ops.equal(where_4, full_10)

        # pd_op.full: (4x49x49xf32) <- ()
        full_35 = paddle._C_ops.full(
            [4, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_5 = paddle._C_ops.where(equal_2, full_35, where_4)
        del equal_2, where_4

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_27 = paddle._C_ops.shape64(reshape_80)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(
            shape64_27, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_27

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_32 = paddle._C_ops.matmul(reshape_80, parameter_88, False, False)
        del parameter_88, reshape_80

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_39 = paddle._C_ops.add(matmul_32, parameter_87)
        del matmul_32, parameter_87

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_39 = [slice_42, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_37 = paddle._C_ops.stack(combine_39, 0)
        del combine_39

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_84 = paddle._C_ops.reshape(add_39, stack_37)
        del add_39, stack_37

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_84, [2, 0, 3, 1, 4])
        del reshape_84

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(
            transpose_35, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_35

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(slice_43, full_7, float("0"), True)
        del slice_43

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_36 = paddle._C_ops.transpose(slice_44, [0, 1, 3, 2])
        del slice_44

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_33 = paddle._C_ops.matmul(scale_5, transpose_36, False, False)
        del scale_5, transpose_36

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_85 = paddle._C_ops.reshape(data_18, full_int_array_7)
        del data_18

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_5 = paddle._C_ops.index_select(data_5, reshape_85, 0)
        del data_5, reshape_85

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_86 = paddle._C_ops.reshape(index_select_5, full_int_array_8)
        del index_select_5

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_86, [2, 0, 1])
        del reshape_86

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(transpose_37, full_int_array_1)
        del transpose_37

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_40 = paddle._C_ops.add(matmul_33, unsqueeze_15)
        del matmul_33, unsqueeze_15

        # pd_op.full: (xi64) <- ()
        full_36 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_2 = paddle._C_ops.floor_divide(slice_42, full_36)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_40 = [floor_divide_2, full_20, full_31, full_4, full_4]
        del floor_divide_2

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_38 = paddle._C_ops.stack(combine_40, 0)
        del combine_40

        # pd_op.reshape: (-1x4x12x49x49xf32) <- (-1x12x49x49xf32, 5xi64)
        reshape_87 = paddle._C_ops.reshape(add_40, stack_38)
        del add_40, stack_38

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(where_5, full_int_array_2)
        del where_5

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(unsqueeze_16, full_int_array_1)
        del unsqueeze_16

        # pd_op.add: (-1x4x12x49x49xf32) <- (-1x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_41 = paddle._C_ops.add(reshape_87, unsqueeze_17)
        del reshape_87, unsqueeze_17

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_41 = [slice_42, full_31, full_4, full_4]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_39 = paddle._C_ops.stack(combine_41, 0)
        del combine_41

        # pd_op.reshape: (-1x12x49x49xf32) <- (-1x4x12x49x49xf32, 4xi64)
        reshape_88 = paddle._C_ops.reshape(add_41, stack_39)
        del add_41, stack_39

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_5 = paddle._C_ops.softmax(reshape_88, -1)
        del reshape_88

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_34 = paddle._C_ops.matmul(softmax_5, slice_45, False, False)
        del slice_45, softmax_5

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])
        del matmul_34

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_42 = [slice_42, full_4, full_17]
        del slice_42

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_40 = paddle._C_ops.stack(combine_42, 0)
        del combine_42

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_89 = paddle._C_ops.reshape(transpose_38, stack_40)
        del stack_40, transpose_38

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_35 = paddle._C_ops.matmul(reshape_89, parameter_86, False, False)
        del parameter_86, reshape_89

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_42 = paddle._C_ops.add(matmul_35, parameter_85)
        del matmul_35, parameter_85

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_90 = paddle._C_ops.reshape(add_42, full_int_array_38)
        del add_42

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_91 = paddle._C_ops.reshape(reshape_90, full_int_array_40)
        del reshape_90

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_39 = paddle._C_ops.transpose(reshape_91, [0, 1, 3, 2, 4, 5])
        del reshape_91

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_92 = paddle._C_ops.reshape(transpose_39, full_int_array_41)
        del transpose_39

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_5 = paddle._C_ops.roll(reshape_92, full_int_array_29, [1, 2])
        del reshape_92

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_43 = [slice_39, full_32, full_17]
        del slice_39

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_41 = paddle._C_ops.stack(combine_43, 0)
        del combine_43

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_93 = paddle._C_ops.reshape(roll_5, stack_41)
        del roll_5, stack_41

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_43 = paddle._C_ops.add(add_38, reshape_93)
        del add_38, reshape_93

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_43, parameter_84, parameter_83, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_83, parameter_84

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_42, parameter_82, False, False)
        del layer_norm_42, parameter_82

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_44 = paddle._C_ops.add(matmul_36, parameter_81)
        del matmul_36, parameter_81

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add_44, False)
        del add_44

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_37 = paddle._C_ops.matmul(gelu_5, parameter_80, False, False)
        del gelu_5, parameter_80

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_45 = paddle._C_ops.add(matmul_37, parameter_79)
        del matmul_37, parameter_79

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_46 = paddle._C_ops.add(add_43, add_45)
        del add_43, add_45

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_28 = paddle._C_ops.shape64(add_46)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(
            shape64_28, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_28

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_46, parameter_78, parameter_77, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_77, parameter_78

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_44 = [slice_46, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_42 = paddle._C_ops.stack(combine_44, 0)
        del combine_44

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_94 = paddle._C_ops.reshape(layer_norm_45, stack_42)
        del layer_norm_45, stack_42

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_29 = paddle._C_ops.shape64(reshape_94)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(
            shape64_29, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_29

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_45 = [slice_47, full_30, full_3, full_30, full_3, full_17]
        del slice_47

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_43 = paddle._C_ops.stack(combine_45, 0)
        del combine_45

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_95 = paddle._C_ops.reshape(reshape_94, stack_43)
        del reshape_94, stack_43

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_95, [0, 1, 3, 2, 4, 5])
        del reshape_95

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_96 = paddle._C_ops.reshape(transpose_40, full_int_array_38)
        del transpose_40

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_97 = paddle._C_ops.reshape(reshape_96, full_int_array_39)
        del reshape_96

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_30 = paddle._C_ops.shape64(reshape_97)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(
            shape64_30, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_30

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_38 = paddle._C_ops.matmul(reshape_97, parameter_76, False, False)
        del parameter_76, reshape_97

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_47 = paddle._C_ops.add(matmul_38, parameter_75)
        del matmul_38, parameter_75

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_46 = [slice_48, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_44 = paddle._C_ops.stack(combine_46, 0)
        del combine_46

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_98 = paddle._C_ops.reshape(add_47, stack_44)
        del add_47, stack_44

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_98, [2, 0, 3, 1, 4])
        del reshape_98

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(
            transpose_41, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_41

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_49, full_7, float("0"), True)
        del slice_49

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_42 = paddle._C_ops.transpose(slice_50, [0, 1, 3, 2])
        del slice_50

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_39 = paddle._C_ops.matmul(scale_6, transpose_42, False, False)
        del scale_6, transpose_42

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_99 = paddle._C_ops.reshape(data_19, full_int_array_7)
        del data_19

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_6 = paddle._C_ops.index_select(data_6, reshape_99, 0)
        del data_6, reshape_99

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_100 = paddle._C_ops.reshape(index_select_6, full_int_array_8)
        del index_select_6

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_100, [2, 0, 1])
        del reshape_100

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(transpose_43, full_int_array_1)
        del transpose_43

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_48 = paddle._C_ops.add(matmul_39, unsqueeze_18)
        del matmul_39, unsqueeze_18

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_6 = paddle._C_ops.softmax(add_48, -1)
        del add_48

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_40 = paddle._C_ops.matmul(softmax_6, slice_51, False, False)
        del slice_51, softmax_6

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_44 = paddle._C_ops.transpose(matmul_40, [0, 2, 1, 3])
        del matmul_40

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_47 = [slice_48, full_4, full_17]
        del slice_48

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_45 = paddle._C_ops.stack(combine_47, 0)
        del combine_47

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_101 = paddle._C_ops.reshape(transpose_44, stack_45)
        del stack_45, transpose_44

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_41 = paddle._C_ops.matmul(reshape_101, parameter_74, False, False)
        del parameter_74, reshape_101

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_49 = paddle._C_ops.add(matmul_41, parameter_73)
        del matmul_41, parameter_73

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_102 = paddle._C_ops.reshape(add_49, full_int_array_38)
        del add_49

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_103 = paddle._C_ops.reshape(reshape_102, full_int_array_40)
        del reshape_102

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_45 = paddle._C_ops.transpose(reshape_103, [0, 1, 3, 2, 4, 5])
        del reshape_103

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_104 = paddle._C_ops.reshape(transpose_45, full_int_array_41)
        del transpose_45

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_48 = [slice_46, full_32, full_17]
        del slice_46

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_46 = paddle._C_ops.stack(combine_48, 0)
        del combine_48

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_105 = paddle._C_ops.reshape(reshape_104, stack_46)
        del reshape_104, stack_46

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_50 = paddle._C_ops.add(add_46, reshape_105)
        del add_46, reshape_105

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_50, parameter_72, parameter_71, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_71, parameter_72

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_48, parameter_70, False, False)
        del layer_norm_48, parameter_70

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_51 = paddle._C_ops.add(matmul_42, parameter_69)
        del matmul_42, parameter_69

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_51, False)
        del add_51

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_43 = paddle._C_ops.matmul(gelu_6, parameter_68, False, False)
        del gelu_6, parameter_68

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_52 = paddle._C_ops.add(matmul_43, parameter_67)
        del matmul_43, parameter_67

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_53 = paddle._C_ops.add(add_50, add_52)
        del add_50, add_52

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_31 = paddle._C_ops.shape64(add_53)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(
            shape64_31, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_31

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_53, parameter_66, parameter_65, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_65, parameter_66

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_49 = [slice_52, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_47 = paddle._C_ops.stack(combine_49, 0)
        del combine_49

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_106 = paddle._C_ops.reshape(layer_norm_51, stack_47)
        del layer_norm_51, stack_47

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_32 = paddle._C_ops.shape64(reshape_106)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(
            shape64_32, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_32

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_6 = paddle._C_ops.roll(reshape_106, full_int_array_11, [1, 2])
        del reshape_106

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_33 = paddle._C_ops.shape64(roll_6)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(
            shape64_33, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_33

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_50 = [slice_54, full_30, full_3, full_30, full_3, full_17]
        del slice_54

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_48 = paddle._C_ops.stack(combine_50, 0)
        del combine_50

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_107 = paddle._C_ops.reshape(roll_6, stack_48)
        del roll_6, stack_48

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_107, [0, 1, 3, 2, 4, 5])
        del reshape_107

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_108 = paddle._C_ops.reshape(transpose_46, full_int_array_38)
        del transpose_46

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_109 = paddle._C_ops.reshape(reshape_108, full_int_array_39)
        del reshape_108

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_37 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__27 = paddle._C_ops.set_value_(
            full_37,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_37

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__28 = paddle._C_ops.set_value_(
            set_value__27,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
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
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
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
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
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
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
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
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
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
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
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
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
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
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__34

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_110 = paddle._C_ops.reshape(set_value__35, full_int_array_42)

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_47 = paddle._C_ops.transpose(reshape_110, [0, 1, 3, 2, 4, 5])
        del reshape_110

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_111 = paddle._C_ops.reshape(transpose_47, full_int_array_27)
        del transpose_47

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_112 = paddle._C_ops.reshape(reshape_111, full_int_array_28)
        del reshape_111

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(reshape_112, full_int_array_2)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(reshape_112, full_int_array_5)
        del reshape_112

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_3 = paddle._C_ops.subtract(unsqueeze_19, unsqueeze_20)
        del unsqueeze_19, unsqueeze_20

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_3 = paddle._C_ops.not_equal(subtract_3, full_10)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_6 = paddle._C_ops.where(not_equal_3, full_34, subtract_3)
        del not_equal_3, subtract_3

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_3 = paddle._C_ops.equal(where_6, full_10)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_7 = paddle._C_ops.where(equal_3, full_35, where_6)
        del equal_3, where_6

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_34 = paddle._C_ops.shape64(reshape_109)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(
            shape64_34, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_34

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_44 = paddle._C_ops.matmul(reshape_109, parameter_64, False, False)
        del parameter_64, reshape_109

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_54 = paddle._C_ops.add(matmul_44, parameter_63)
        del matmul_44, parameter_63

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_51 = [slice_55, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_49 = paddle._C_ops.stack(combine_51, 0)
        del combine_51

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_113 = paddle._C_ops.reshape(add_54, stack_49)
        del add_54, stack_49

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_48 = paddle._C_ops.transpose(reshape_113, [2, 0, 3, 1, 4])
        del reshape_113

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(
            transpose_48, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_48

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_56, full_7, float("0"), True)
        del slice_56

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_49 = paddle._C_ops.transpose(slice_57, [0, 1, 3, 2])
        del slice_57

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_45 = paddle._C_ops.matmul(scale_7, transpose_49, False, False)
        del scale_7, transpose_49

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_114 = paddle._C_ops.reshape(data_20, full_int_array_7)
        del data_20

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_7 = paddle._C_ops.index_select(data_7, reshape_114, 0)
        del data_7, reshape_114

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_115 = paddle._C_ops.reshape(index_select_7, full_int_array_8)
        del index_select_7

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_115, [2, 0, 1])
        del reshape_115

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(transpose_50, full_int_array_1)
        del transpose_50

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_55 = paddle._C_ops.add(matmul_45, unsqueeze_21)
        del matmul_45, unsqueeze_21

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_3 = paddle._C_ops.floor_divide(slice_55, full_36)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_52 = [floor_divide_3, full_20, full_31, full_4, full_4]
        del floor_divide_3

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_50 = paddle._C_ops.stack(combine_52, 0)
        del combine_52

        # pd_op.reshape: (-1x4x12x49x49xf32) <- (-1x12x49x49xf32, 5xi64)
        reshape_116 = paddle._C_ops.reshape(add_55, stack_50)
        del add_55, stack_50

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(where_7, full_int_array_2)
        del where_7

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(unsqueeze_22, full_int_array_1)
        del unsqueeze_22

        # pd_op.add: (-1x4x12x49x49xf32) <- (-1x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_56 = paddle._C_ops.add(reshape_116, unsqueeze_23)
        del reshape_116, unsqueeze_23

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_53 = [slice_55, full_31, full_4, full_4]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_51 = paddle._C_ops.stack(combine_53, 0)
        del combine_53

        # pd_op.reshape: (-1x12x49x49xf32) <- (-1x4x12x49x49xf32, 4xi64)
        reshape_117 = paddle._C_ops.reshape(add_56, stack_51)
        del add_56, stack_51

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_7 = paddle._C_ops.softmax(reshape_117, -1)
        del reshape_117

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_46 = paddle._C_ops.matmul(softmax_7, slice_58, False, False)
        del slice_58, softmax_7

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_51 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_54 = [slice_55, full_4, full_17]
        del slice_55

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_52 = paddle._C_ops.stack(combine_54, 0)
        del combine_54

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_118 = paddle._C_ops.reshape(transpose_51, stack_52)
        del stack_52, transpose_51

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_118, parameter_62, False, False)
        del parameter_62, reshape_118

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_57 = paddle._C_ops.add(matmul_47, parameter_61)
        del matmul_47, parameter_61

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_119 = paddle._C_ops.reshape(add_57, full_int_array_38)
        del add_57

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_120 = paddle._C_ops.reshape(reshape_119, full_int_array_40)
        del reshape_119

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_120, [0, 1, 3, 2, 4, 5])
        del reshape_120

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_121 = paddle._C_ops.reshape(transpose_52, full_int_array_41)
        del transpose_52

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_7 = paddle._C_ops.roll(reshape_121, full_int_array_29, [1, 2])
        del reshape_121

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_55 = [slice_52, full_32, full_17]
        del slice_52

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_53 = paddle._C_ops.stack(combine_55, 0)
        del combine_55

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_122 = paddle._C_ops.reshape(roll_7, stack_53)
        del roll_7, stack_53

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_58 = paddle._C_ops.add(add_53, reshape_122)
        del add_53, reshape_122

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_58, parameter_60, parameter_59, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_59, parameter_60

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_54, parameter_58, False, False)
        del layer_norm_54, parameter_58

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_59 = paddle._C_ops.add(matmul_48, parameter_57)
        del matmul_48, parameter_57

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_59, False)
        del add_59

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_7, parameter_56, False, False)
        del gelu_7, parameter_56

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_60 = paddle._C_ops.add(matmul_49, parameter_55)
        del matmul_49, parameter_55

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_61 = paddle._C_ops.add(add_58, add_60)
        del add_58, add_60

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_35 = paddle._C_ops.shape64(add_61)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(
            shape64_35, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_35

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_61, parameter_54, parameter_53, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_53, parameter_54

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_56 = [slice_59, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_54 = paddle._C_ops.stack(combine_56, 0)
        del combine_56

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_123 = paddle._C_ops.reshape(layer_norm_57, stack_54)
        del layer_norm_57, stack_54

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_36 = paddle._C_ops.shape64(reshape_123)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(
            shape64_36, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_36

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_57 = [slice_60, full_30, full_3, full_30, full_3, full_17]
        del slice_60

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_55 = paddle._C_ops.stack(combine_57, 0)
        del combine_57

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_124 = paddle._C_ops.reshape(reshape_123, stack_55)
        del reshape_123, stack_55

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_53 = paddle._C_ops.transpose(reshape_124, [0, 1, 3, 2, 4, 5])
        del reshape_124

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_125 = paddle._C_ops.reshape(transpose_53, full_int_array_38)
        del transpose_53

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_126 = paddle._C_ops.reshape(reshape_125, full_int_array_39)
        del reshape_125

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_37 = paddle._C_ops.shape64(reshape_126)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(
            shape64_37, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_37

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_50 = paddle._C_ops.matmul(reshape_126, parameter_52, False, False)
        del parameter_52, reshape_126

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_62 = paddle._C_ops.add(matmul_50, parameter_51)
        del matmul_50, parameter_51

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_58 = [slice_61, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_56 = paddle._C_ops.stack(combine_58, 0)
        del combine_58

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_127 = paddle._C_ops.reshape(add_62, stack_56)
        del add_62, stack_56

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_127, [2, 0, 3, 1, 4])
        del reshape_127

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(
            transpose_54, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_54

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(slice_62, full_7, float("0"), True)
        del slice_62

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_55 = paddle._C_ops.transpose(slice_63, [0, 1, 3, 2])
        del slice_63

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_51 = paddle._C_ops.matmul(scale_8, transpose_55, False, False)
        del scale_8, transpose_55

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_128 = paddle._C_ops.reshape(data_21, full_int_array_7)
        del data_21

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_8 = paddle._C_ops.index_select(data_8, reshape_128, 0)
        del data_8, reshape_128

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_129 = paddle._C_ops.reshape(index_select_8, full_int_array_8)
        del index_select_8

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_129, [2, 0, 1])
        del reshape_129

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(transpose_56, full_int_array_1)
        del transpose_56

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_63 = paddle._C_ops.add(matmul_51, unsqueeze_24)
        del matmul_51, unsqueeze_24

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_8 = paddle._C_ops.softmax(add_63, -1)
        del add_63

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_52 = paddle._C_ops.matmul(softmax_8, slice_64, False, False)
        del slice_64, softmax_8

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_59 = [slice_61, full_4, full_17]
        del slice_61

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_57 = paddle._C_ops.stack(combine_59, 0)
        del combine_59

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_130 = paddle._C_ops.reshape(transpose_57, stack_57)
        del stack_57, transpose_57

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_130, parameter_50, False, False)
        del parameter_50, reshape_130

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_64 = paddle._C_ops.add(matmul_53, parameter_49)
        del matmul_53, parameter_49

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_131 = paddle._C_ops.reshape(add_64, full_int_array_38)
        del add_64

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_132 = paddle._C_ops.reshape(reshape_131, full_int_array_40)
        del reshape_131

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_132, [0, 1, 3, 2, 4, 5])
        del reshape_132

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_133 = paddle._C_ops.reshape(transpose_58, full_int_array_41)
        del transpose_58

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_60 = [slice_59, full_32, full_17]
        del slice_59

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_58 = paddle._C_ops.stack(combine_60, 0)
        del combine_60

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_134 = paddle._C_ops.reshape(reshape_133, stack_58)
        del reshape_133, stack_58

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_65 = paddle._C_ops.add(add_61, reshape_134)
        del add_61, reshape_134

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_48, parameter_47, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_47, parameter_48

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_60, parameter_46, False, False)
        del layer_norm_60, parameter_46

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_66 = paddle._C_ops.add(matmul_54, parameter_45)
        del matmul_54, parameter_45

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_66, False)
        del add_66

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_55 = paddle._C_ops.matmul(gelu_8, parameter_44, False, False)
        del gelu_8, parameter_44

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_67 = paddle._C_ops.add(matmul_55, parameter_43)
        del matmul_55, parameter_43

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_68 = paddle._C_ops.add(add_65, add_67)
        del add_65, add_67

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_38 = paddle._C_ops.shape64(add_68)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(
            shape64_38, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_38

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_68, parameter_42, parameter_41, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_41, parameter_42

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_61 = [slice_65, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_59 = paddle._C_ops.stack(combine_61, 0)
        del combine_61

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_135 = paddle._C_ops.reshape(layer_norm_63, stack_59)
        del layer_norm_63, stack_59

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_39 = paddle._C_ops.shape64(reshape_135)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(
            shape64_39, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_39

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_8 = paddle._C_ops.roll(reshape_135, full_int_array_11, [1, 2])
        del reshape_135

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_40 = paddle._C_ops.shape64(roll_8)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(
            shape64_40, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_40

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_62 = [slice_67, full_30, full_3, full_30, full_3, full_17]
        del full_30, slice_67

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_60 = paddle._C_ops.stack(combine_62, 0)
        del combine_62

        # pd_op.reshape: (-1x2x7x2x7x384xf32) <- (-1x14x14x384xf32, 6xi64)
        reshape_136 = paddle._C_ops.reshape(roll_8, stack_60)
        del roll_8, stack_60

        # pd_op.transpose: (-1x2x2x7x7x384xf32) <- (-1x2x7x2x7x384xf32)
        transpose_59 = paddle._C_ops.transpose(reshape_136, [0, 1, 3, 2, 4, 5])
        del reshape_136

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x2x2x7x7x384xf32, 4xi64)
        reshape_137 = paddle._C_ops.reshape(transpose_59, full_int_array_38)
        del transpose_59

        # pd_op.reshape: (-1x49x384xf32) <- (-1x7x7x384xf32, 3xi64)
        reshape_138 = paddle._C_ops.reshape(reshape_137, full_int_array_39)
        del full_int_array_39, reshape_137

        # pd_op.full: (1x14x14x1xf32) <- ()
        full_38 = paddle._C_ops.full(
            [1, 14, 14, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__36 = paddle._C_ops.set_value_(
            full_38,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_38

        # pd_op.set_value_: (1x14x14x1xf32) <- (1x14x14x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__37 = paddle._C_ops.set_value_(
            set_value__36,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
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
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
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
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
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
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
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
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
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
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
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
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
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
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del set_value__43

        # pd_op.reshape: (1x2x7x2x7x1xf32) <- (1x14x14x1xf32, 6xi64)
        reshape_139 = paddle._C_ops.reshape(set_value__44, full_int_array_42)
        del full_int_array_42

        # pd_op.transpose: (1x2x2x7x7x1xf32) <- (1x2x7x2x7x1xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_139, [0, 1, 3, 2, 4, 5])
        del reshape_139

        # pd_op.reshape: (4x7x7x1xf32) <- (1x2x2x7x7x1xf32, 4xi64)
        reshape_140 = paddle._C_ops.reshape(transpose_60, full_int_array_27)
        del transpose_60

        # pd_op.reshape: (4x49xf32) <- (4x7x7x1xf32, 2xi64)
        reshape_141 = paddle._C_ops.reshape(reshape_140, full_int_array_28)
        del reshape_140

        # pd_op.unsqueeze: (4x1x49xf32) <- (4x49xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(reshape_141, full_int_array_2)

        # pd_op.unsqueeze: (4x49x1xf32) <- (4x49xf32, 1xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(reshape_141, full_int_array_5)
        del reshape_141

        # pd_op.subtract: (4x49x49xf32) <- (4x1x49xf32, 4x49x1xf32)
        subtract_4 = paddle._C_ops.subtract(unsqueeze_25, unsqueeze_26)
        del unsqueeze_25, unsqueeze_26

        # pd_op.not_equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        not_equal_4 = paddle._C_ops.not_equal(subtract_4, full_10)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_8 = paddle._C_ops.where(not_equal_4, full_34, subtract_4)
        del full_34, not_equal_4, subtract_4

        # pd_op.equal: (4x49x49xb) <- (4x49x49xf32, xf32)
        equal_4 = paddle._C_ops.equal(where_8, full_10)

        # pd_op.where: (4x49x49xf32) <- (4x49x49xb, 4x49x49xf32, 4x49x49xf32)
        where_9 = paddle._C_ops.where(equal_4, full_35, where_8)
        del equal_4, full_35, where_8

        # pd_op.shape64: (3xi64) <- (-1x49x384xf32)
        shape64_41 = paddle._C_ops.shape64(reshape_138)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(
            shape64_41, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_41

        # pd_op.matmul: (-1x49x1152xf32) <- (-1x49x384xf32, 384x1152xf32)
        matmul_56 = paddle._C_ops.matmul(reshape_138, parameter_40, False, False)
        del parameter_40, reshape_138

        # pd_op.add: (-1x49x1152xf32) <- (-1x49x1152xf32, 1152xf32)
        add_69 = paddle._C_ops.add(matmul_56, parameter_39)
        del matmul_56, parameter_39

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_63 = [slice_68, full_4, full_5, full_31, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_61 = paddle._C_ops.stack(combine_63, 0)
        del combine_63

        # pd_op.reshape: (-1x49x3x12x32xf32) <- (-1x49x1152xf32, 5xi64)
        reshape_142 = paddle._C_ops.reshape(add_69, stack_61)
        del add_69, stack_61

        # pd_op.transpose: (3x-1x12x49x32xf32) <- (-1x49x3x12x32xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_142, [2, 0, 3, 1, 4])
        del reshape_142

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x12x49x32xf32) <- (3x-1x12x49x32xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(
            transpose_61, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_61

        # pd_op.scale: (-1x12x49x32xf32) <- (-1x12x49x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(slice_69, full_7, float("0"), True)
        del slice_69

        # pd_op.transpose: (-1x12x32x49xf32) <- (-1x12x49x32xf32)
        transpose_62 = paddle._C_ops.transpose(slice_70, [0, 1, 3, 2])
        del slice_70

        # pd_op.matmul: (-1x12x49x49xf32) <- (-1x12x49x32xf32, -1x12x32x49xf32)
        matmul_57 = paddle._C_ops.matmul(scale_9, transpose_62, False, False)
        del scale_9, transpose_62

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_143 = paddle._C_ops.reshape(data_22, full_int_array_7)
        del data_22

        # pd_op.index_select: (2401x12xf32) <- (169x12xf32, 2401xi64)
        index_select_9 = paddle._C_ops.index_select(data_9, reshape_143, 0)
        del data_9, reshape_143

        # pd_op.reshape: (49x49x12xf32) <- (2401x12xf32, 3xi64)
        reshape_144 = paddle._C_ops.reshape(index_select_9, full_int_array_8)
        del index_select_9

        # pd_op.transpose: (12x49x49xf32) <- (49x49x12xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_144, [2, 0, 1])
        del reshape_144

        # pd_op.unsqueeze: (1x12x49x49xf32) <- (12x49x49xf32, 1xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(transpose_63, full_int_array_1)
        del transpose_63

        # pd_op.add: (-1x12x49x49xf32) <- (-1x12x49x49xf32, 1x12x49x49xf32)
        add_70 = paddle._C_ops.add(matmul_57, unsqueeze_27)
        del matmul_57, unsqueeze_27

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_4 = paddle._C_ops.floor_divide(slice_68, full_36)
        del full_36

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_64 = [floor_divide_4, full_20, full_31, full_4, full_4]
        del floor_divide_4, full_20

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_62 = paddle._C_ops.stack(combine_64, 0)
        del combine_64

        # pd_op.reshape: (-1x4x12x49x49xf32) <- (-1x12x49x49xf32, 5xi64)
        reshape_145 = paddle._C_ops.reshape(add_70, stack_62)
        del add_70, stack_62

        # pd_op.unsqueeze: (4x1x49x49xf32) <- (4x49x49xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(where_9, full_int_array_2)
        del where_9

        # pd_op.unsqueeze: (1x4x1x49x49xf32) <- (4x1x49x49xf32, 1xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(unsqueeze_28, full_int_array_1)
        del unsqueeze_28

        # pd_op.add: (-1x4x12x49x49xf32) <- (-1x4x12x49x49xf32, 1x4x1x49x49xf32)
        add_71 = paddle._C_ops.add(reshape_145, unsqueeze_29)
        del reshape_145, unsqueeze_29

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_65 = [slice_68, full_31, full_4, full_4]
        del full_31

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_63 = paddle._C_ops.stack(combine_65, 0)
        del combine_65

        # pd_op.reshape: (-1x12x49x49xf32) <- (-1x4x12x49x49xf32, 4xi64)
        reshape_146 = paddle._C_ops.reshape(add_71, stack_63)
        del add_71, stack_63

        # pd_op.softmax: (-1x12x49x49xf32) <- (-1x12x49x49xf32)
        softmax_9 = paddle._C_ops.softmax(reshape_146, -1)
        del reshape_146

        # pd_op.matmul: (-1x12x49x32xf32) <- (-1x12x49x49xf32, -1x12x49x32xf32)
        matmul_58 = paddle._C_ops.matmul(softmax_9, slice_71, False, False)
        del slice_71, softmax_9

        # pd_op.transpose: (-1x49x12x32xf32) <- (-1x12x49x32xf32)
        transpose_64 = paddle._C_ops.transpose(matmul_58, [0, 2, 1, 3])
        del matmul_58

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_66 = [slice_68, full_4, full_17]
        del slice_68

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_64 = paddle._C_ops.stack(combine_66, 0)
        del combine_66

        # pd_op.reshape: (-1x49x384xf32) <- (-1x49x12x32xf32, 3xi64)
        reshape_147 = paddle._C_ops.reshape(transpose_64, stack_64)
        del stack_64, transpose_64

        # pd_op.matmul: (-1x49x384xf32) <- (-1x49x384xf32, 384x384xf32)
        matmul_59 = paddle._C_ops.matmul(reshape_147, parameter_38, False, False)
        del parameter_38, reshape_147

        # pd_op.add: (-1x49x384xf32) <- (-1x49x384xf32, 384xf32)
        add_72 = paddle._C_ops.add(matmul_59, parameter_37)
        del matmul_59, parameter_37

        # pd_op.reshape: (-1x7x7x384xf32) <- (-1x49x384xf32, 4xi64)
        reshape_148 = paddle._C_ops.reshape(add_72, full_int_array_38)
        del add_72, full_int_array_38

        # pd_op.reshape: (-1x2x2x7x7x384xf32) <- (-1x7x7x384xf32, 6xi64)
        reshape_149 = paddle._C_ops.reshape(reshape_148, full_int_array_40)
        del full_int_array_40, reshape_148

        # pd_op.transpose: (-1x2x7x2x7x384xf32) <- (-1x2x2x7x7x384xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_149, [0, 1, 3, 2, 4, 5])
        del reshape_149

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x2x7x2x7x384xf32, 4xi64)
        reshape_150 = paddle._C_ops.reshape(transpose_65, full_int_array_41)
        del full_int_array_41, transpose_65

        # pd_op.roll: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 2xi64)
        roll_9 = paddle._C_ops.roll(reshape_150, full_int_array_29, [1, 2])
        del reshape_150

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_67 = [slice_65, full_32, full_17]
        del full_32, slice_65

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_65 = paddle._C_ops.stack(combine_67, 0)
        del combine_67

        # pd_op.reshape: (-1x196x384xf32) <- (-1x14x14x384xf32, 3xi64)
        reshape_151 = paddle._C_ops.reshape(roll_9, stack_65)
        del roll_9, stack_65

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_73 = paddle._C_ops.add(add_68, reshape_151)
        del add_68, reshape_151

        # pd_op.layer_norm: (-1x196x384xf32, -1x196xf32, -1x196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_73, parameter_36, parameter_35, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_35, parameter_36

        # pd_op.matmul: (-1x196x1536xf32) <- (-1x196x384xf32, 384x1536xf32)
        matmul_60 = paddle._C_ops.matmul(layer_norm_66, parameter_34, False, False)
        del layer_norm_66, parameter_34

        # pd_op.add: (-1x196x1536xf32) <- (-1x196x1536xf32, 1536xf32)
        add_74 = paddle._C_ops.add(matmul_60, parameter_33)
        del matmul_60, parameter_33

        # pd_op.gelu: (-1x196x1536xf32) <- (-1x196x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_74, False)
        del add_74

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x1536xf32, 1536x384xf32)
        matmul_61 = paddle._C_ops.matmul(gelu_9, parameter_32, False, False)
        del gelu_9, parameter_32

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add_75 = paddle._C_ops.add(matmul_61, parameter_31)
        del matmul_61, parameter_31

        # pd_op.add: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add_76 = paddle._C_ops.add(add_73, add_75)
        del add_73, add_75

        # pd_op.shape64: (3xi64) <- (-1x196x384xf32)
        shape64_42 = paddle._C_ops.shape64(add_76)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(
            shape64_42, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_42

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_68 = [slice_72, full_29, full_29, full_17]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_66 = paddle._C_ops.stack(combine_68, 0)
        del combine_68

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x196x384xf32, 4xi64)
        reshape_152 = paddle._C_ops.reshape(add_76, stack_66)
        del add_76, stack_66

        # pd_op.strided_slice: (-1x7x7x384xf32) <- (-1x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_8 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_12, full_int_array_25, full_int_array_30
        )

        # pd_op.strided_slice: (-1x7x7x384xf32) <- (-1x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_9 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_31, full_int_array_25, full_int_array_30
        )
        del full_int_array_31

        # pd_op.strided_slice: (-1x7x7x384xf32) <- (-1x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_10 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_32, full_int_array_25, full_int_array_30
        )
        del full_int_array_32

        # pd_op.strided_slice: (-1x7x7x384xf32) <- (-1x14x14x384xf32, 2xi64, 2xi64, 2xi64)
        strided_slice_11 = paddle._C_ops.strided_slice(
            reshape_152, [1, 2], full_int_array_14, full_int_array_25, full_int_array_30
        )
        del full_int_array_30

        # pd_op.shape64: (4xi64) <- (-1x14x14x384xf32)
        shape64_43 = paddle._C_ops.shape64(reshape_152)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(
            shape64_43, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_43

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_69 = [slice_73, full_29, full_29, full_17]
        del full_17, full_29, slice_73

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_67 = paddle._C_ops.stack(combine_69, 0)
        del combine_69

        # pd_op.reshape: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 4xi64)
        reshape_153 = paddle._C_ops.reshape(reshape_152, stack_67)
        del reshape_152, stack_67

        # builtin.combine: ([-1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32]) <- (-1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32)
        combine_70 = [
            strided_slice_8,
            strided_slice_9,
            strided_slice_10,
            strided_slice_11,
        ]
        del strided_slice_10, strided_slice_11, strided_slice_8, strided_slice_9

        # pd_op.concat: (-1x7x7x1536xf32) <- ([-1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32, -1x7x7x384xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_70, full_15)
        del combine_70, full_15

        # pd_op.full: (xi64) <- ()
        full_39 = paddle._C_ops.full(
            [], float("1536"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_71 = [slice_72, full_16, full_39]
        del full_16, full_39, slice_72

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_68 = paddle._C_ops.stack(combine_71, 0)
        del combine_71

        # pd_op.reshape: (-1x-1x1536xf32) <- (-1x7x7x1536xf32, 3xi64)
        reshape_154 = paddle._C_ops.reshape(concat_2, stack_68)
        del concat_2, stack_68

        # pd_op.layer_norm: (-1x-1x1536xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x1536xf32, 1536xf32, 1536xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                reshape_154, parameter_30, parameter_29, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_29, parameter_30, reshape_154

        # pd_op.matmul: (-1x-1x768xf32) <- (-1x-1x1536xf32, 1536x768xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_69, parameter_28, False, False)
        del layer_norm_69, parameter_28

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        shape64_44 = paddle._C_ops.shape64(matmul_62)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(
            shape64_44, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_44

        # pd_op.shape64: (3xi64) <- (-1x-1x768xf32)
        shape64_45 = paddle._C_ops.shape64(matmul_62)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(
            shape64_45, [0], full_int_array_2, full_int_array_5, [1], [0]
        )
        del shape64_45

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                matmul_62, parameter_27, parameter_26, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_26, parameter_27

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_72 = [slice_74, full_3, full_3, full_28]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_69 = paddle._C_ops.stack(combine_72, 0)
        del combine_72

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x-1x768xf32, 4xi64)
        reshape_155 = paddle._C_ops.reshape(layer_norm_72, stack_69)
        del layer_norm_72, stack_69

        # pd_op.shape64: (4xi64) <- (-1x7x7x768xf32)
        shape64_46 = paddle._C_ops.shape64(reshape_155)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(
            shape64_46, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_46

        # pd_op.full: (xi64) <- ()
        full_40 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_73 = [slice_76, full_40, full_3, full_40, full_3, full_28]
        del slice_76

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_70 = paddle._C_ops.stack(combine_73, 0)
        del combine_73

        # pd_op.reshape: (-1x1x7x1x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        reshape_156 = paddle._C_ops.reshape(reshape_155, stack_70)
        del reshape_155, stack_70

        # pd_op.transpose: (-1x1x1x7x7x768xf32) <- (-1x1x7x1x7x768xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_156, [0, 1, 3, 2, 4, 5])
        del reshape_156

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [-1, 7, 7, 768]

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x1x1x7x7x768xf32, 4xi64)
        reshape_157 = paddle._C_ops.reshape(transpose_66, full_int_array_43)
        del transpose_66

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_44 = [-1, 49, 768]

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        reshape_158 = paddle._C_ops.reshape(reshape_157, full_int_array_44)
        del reshape_157

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        shape64_47 = paddle._C_ops.shape64(reshape_158)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(
            shape64_47, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_47

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        matmul_63 = paddle._C_ops.matmul(reshape_158, parameter_25, False, False)
        del parameter_25, reshape_158

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        add_77 = paddle._C_ops.add(matmul_63, parameter_24)
        del matmul_63, parameter_24

        # pd_op.full: (xi64) <- ()
        full_41 = paddle._C_ops.full(
            [], float("24"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_74 = [slice_77, full_4, full_5, full_41, full_6]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_71 = paddle._C_ops.stack(combine_74, 0)
        del combine_74

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        reshape_159 = paddle._C_ops.reshape(add_77, stack_71)
        del add_77, stack_71

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        transpose_67 = paddle._C_ops.transpose(reshape_159, [2, 0, 3, 1, 4])
        del reshape_159

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(
            transpose_67, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del transpose_67

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(slice_78, full_7, float("0"), True)
        del slice_78

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        transpose_68 = paddle._C_ops.transpose(slice_79, [0, 1, 3, 2])
        del slice_79

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        matmul_64 = paddle._C_ops.matmul(scale_10, transpose_68, False, False)
        del scale_10, transpose_68

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_160 = paddle._C_ops.reshape(data_23, full_int_array_7)
        del data_23

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_10 = paddle._C_ops.index_select(data_10, reshape_160, 0)
        del data_10, reshape_160

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_161 = paddle._C_ops.reshape(index_select_10, full_int_array_8)
        del index_select_10

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_69 = paddle._C_ops.transpose(reshape_161, [2, 0, 1])
        del reshape_161

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(transpose_69, full_int_array_1)
        del transpose_69

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        add_78 = paddle._C_ops.add(matmul_64, unsqueeze_30)
        del matmul_64, unsqueeze_30

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        softmax_10 = paddle._C_ops.softmax(add_78, -1)
        del add_78

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        matmul_65 = paddle._C_ops.matmul(softmax_10, slice_80, False, False)
        del slice_80, softmax_10

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        transpose_70 = paddle._C_ops.transpose(matmul_65, [0, 2, 1, 3])
        del matmul_65

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_75 = [slice_77, full_4, full_28]
        del slice_77

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_72 = paddle._C_ops.stack(combine_75, 0)
        del combine_75

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        reshape_162 = paddle._C_ops.reshape(transpose_70, stack_72)
        del stack_72, transpose_70

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        matmul_66 = paddle._C_ops.matmul(reshape_162, parameter_23, False, False)
        del parameter_23, reshape_162

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        add_79 = paddle._C_ops.add(matmul_66, parameter_22)
        del matmul_66, parameter_22

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        reshape_163 = paddle._C_ops.reshape(add_79, full_int_array_43)
        del add_79

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_45 = [-1, 1, 1, 7, 7, 768]

        # pd_op.reshape: (-1x1x1x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        reshape_164 = paddle._C_ops.reshape(reshape_163, full_int_array_45)
        del reshape_163

        # pd_op.transpose: (-1x1x7x1x7x768xf32) <- (-1x1x1x7x7x768xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_164, [0, 1, 3, 2, 4, 5])
        del reshape_164

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x1x7x1x7x768xf32, 4xi64)
        reshape_165 = paddle._C_ops.reshape(transpose_71, full_int_array_43)
        del transpose_71

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_76 = [slice_74, full_4, full_28]
        del slice_74

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_73 = paddle._C_ops.stack(combine_76, 0)
        del combine_76

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        reshape_166 = paddle._C_ops.reshape(reshape_165, stack_73)
        del reshape_165, stack_73

        # pd_op.add: (-1x49x768xf32) <- (-1x-1x768xf32, -1x49x768xf32)
        add_80 = paddle._C_ops.add(matmul_62, reshape_166)
        del matmul_62, reshape_166

        # pd_op.layer_norm: (-1x49x768xf32, -1x49xf32, -1x49xf32) <- (-1x49x768xf32, 768xf32, 768xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_21, parameter_20, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_20, parameter_21

        # pd_op.matmul: (-1x49x3072xf32) <- (-1x49x768xf32, 768x3072xf32)
        matmul_67 = paddle._C_ops.matmul(layer_norm_75, parameter_19, False, False)
        del layer_norm_75, parameter_19

        # pd_op.add: (-1x49x3072xf32) <- (-1x49x3072xf32, 3072xf32)
        add_81 = paddle._C_ops.add(matmul_67, parameter_18)
        del matmul_67, parameter_18

        # pd_op.gelu: (-1x49x3072xf32) <- (-1x49x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_81, False)
        del add_81

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x3072xf32, 3072x768xf32)
        matmul_68 = paddle._C_ops.matmul(gelu_10, parameter_17, False, False)
        del gelu_10, parameter_17

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        add_82 = paddle._C_ops.add(matmul_68, parameter_16)
        del matmul_68, parameter_16

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, -1x49x768xf32)
        add_83 = paddle._C_ops.add(add_80, add_82)
        del add_80, add_82

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        shape64_48 = paddle._C_ops.shape64(add_83)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(
            shape64_48, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_48

        # pd_op.layer_norm: (-1x49x768xf32, -1x49xf32, -1x49xf32) <- (-1x49x768xf32, 768xf32, 768xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_83, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_14, parameter_15

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_77 = [slice_81, full_3, full_3, full_28]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_74 = paddle._C_ops.stack(combine_77, 0)
        del combine_77

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        reshape_167 = paddle._C_ops.reshape(layer_norm_78, stack_74)
        del layer_norm_78, stack_74

        # pd_op.shape64: (4xi64) <- (-1x7x7x768xf32)
        shape64_49 = paddle._C_ops.shape64(reshape_167)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(
            shape64_49, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_49

        # pd_op.roll: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 2xi64)
        roll_10 = paddle._C_ops.roll(reshape_167, full_int_array_11, [1, 2])
        del reshape_167

        # pd_op.shape64: (4xi64) <- (-1x7x7x768xf32)
        shape64_50 = paddle._C_ops.shape64(roll_10)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(
            shape64_50, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_50

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_78 = [slice_83, full_40, full_3, full_40, full_3, full_28]
        del full_3, slice_83

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_75 = paddle._C_ops.stack(combine_78, 0)
        del combine_78

        # pd_op.reshape: (-1x1x7x1x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        reshape_168 = paddle._C_ops.reshape(roll_10, stack_75)
        del roll_10, stack_75

        # pd_op.transpose: (-1x1x1x7x7x768xf32) <- (-1x1x7x1x7x768xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_168, [0, 1, 3, 2, 4, 5])
        del reshape_168

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x1x1x7x7x768xf32, 4xi64)
        reshape_169 = paddle._C_ops.reshape(transpose_72, full_int_array_43)
        del transpose_72

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        reshape_170 = paddle._C_ops.reshape(reshape_169, full_int_array_44)
        del full_int_array_44, reshape_169

        # pd_op.full: (1x7x7x1xf32) <- ()
        full_42 = paddle._C_ops.full(
            [1, 7, 7, 1],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__45 = paddle._C_ops.set_value_(
            full_42,
            full_int_array_12,
            full_int_array_13,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_42, full_int_array_12

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__46 = paddle._C_ops.set_value_(
            set_value__45,
            full_int_array_15,
            full_int_array_16,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_15, set_value__45

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__47 = paddle._C_ops.set_value_(
            set_value__46,
            full_int_array_17,
            full_int_array_18,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_17, full_int_array_18, set_value__46

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__48 = paddle._C_ops.set_value_(
            set_value__47,
            full_int_array_19,
            full_int_array_20,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del full_int_array_19, set_value__47

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__49 = paddle._C_ops.set_value_(
            set_value__48,
            full_int_array_13,
            full_int_array_11,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del full_int_array_13, set_value__48

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__50 = paddle._C_ops.set_value_(
            set_value__49,
            full_int_array_16,
            full_int_array_21,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del full_int_array_16, full_int_array_21, set_value__49

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__51 = paddle._C_ops.set_value_(
            set_value__50,
            full_int_array_22,
            full_int_array_23,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del full_int_array_22, full_int_array_23, set_value__50

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__52 = paddle._C_ops.set_value_(
            set_value__51,
            full_int_array_20,
            full_int_array_24,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del full_int_array_20, full_int_array_24, set_value__51

        # pd_op.set_value_: (1x7x7x1xf32) <- (1x7x7x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__53 = paddle._C_ops.set_value_(
            set_value__52,
            full_int_array_11,
            full_int_array_25,
            full_int_array_14,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del full_int_array_11, full_int_array_25, set_value__52

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_46 = [1, 1, 7, 1, 7, 1]

        # pd_op.reshape: (1x1x7x1x7x1xf32) <- (1x7x7x1xf32, 6xi64)
        reshape_171 = paddle._C_ops.reshape(set_value__53, full_int_array_46)
        del full_int_array_46

        # pd_op.transpose: (1x1x1x7x7x1xf32) <- (1x1x7x1x7x1xf32)
        transpose_73 = paddle._C_ops.transpose(reshape_171, [0, 1, 3, 2, 4, 5])
        del reshape_171

        # pd_op.reshape: (1x7x7x1xf32) <- (1x1x1x7x7x1xf32, 4xi64)
        reshape_172 = paddle._C_ops.reshape(transpose_73, full_int_array_27)
        del full_int_array_27, transpose_73

        # pd_op.reshape: (1x49xf32) <- (1x7x7x1xf32, 2xi64)
        reshape_173 = paddle._C_ops.reshape(reshape_172, full_int_array_28)
        del full_int_array_28, reshape_172

        # pd_op.unsqueeze: (1x1x49xf32) <- (1x49xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(reshape_173, full_int_array_2)

        # pd_op.unsqueeze: (1x49x1xf32) <- (1x49xf32, 1xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(reshape_173, full_int_array_5)
        del reshape_173

        # pd_op.subtract: (1x49x49xf32) <- (1x1x49xf32, 1x49x1xf32)
        subtract_5 = paddle._C_ops.subtract(unsqueeze_31, unsqueeze_32)
        del unsqueeze_31, unsqueeze_32

        # pd_op.not_equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        not_equal_5 = paddle._C_ops.not_equal(subtract_5, full_10)

        # pd_op.full: (1x49x49xf32) <- ()
        full_43 = paddle._C_ops.full(
            [1, 49, 49],
            float("-100"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        where_10 = paddle._C_ops.where(not_equal_5, full_43, subtract_5)
        del full_43, not_equal_5, subtract_5

        # pd_op.equal: (1x49x49xb) <- (1x49x49xf32, xf32)
        equal_5 = paddle._C_ops.equal(where_10, full_10)
        del full_10

        # pd_op.full: (1x49x49xf32) <- ()
        full_44 = paddle._C_ops.full(
            [1, 49, 49],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.where: (1x49x49xf32) <- (1x49x49xb, 1x49x49xf32, 1x49x49xf32)
        where_11 = paddle._C_ops.where(equal_5, full_44, where_10)
        del equal_5, full_44, where_10

        # pd_op.shape64: (3xi64) <- (-1x49x768xf32)
        shape64_51 = paddle._C_ops.shape64(reshape_170)

        # pd_op.slice: (xi64) <- (3xi64, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(
            shape64_51, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_51

        # pd_op.matmul: (-1x49x2304xf32) <- (-1x49x768xf32, 768x2304xf32)
        matmul_69 = paddle._C_ops.matmul(reshape_170, parameter_13, False, False)
        del parameter_13, reshape_170

        # pd_op.add: (-1x49x2304xf32) <- (-1x49x2304xf32, 2304xf32)
        add_84 = paddle._C_ops.add(matmul_69, parameter_12)
        del matmul_69, parameter_12

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_79 = [slice_84, full_4, full_5, full_41, full_6]
        del full_5, full_6

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_76 = paddle._C_ops.stack(combine_79, 0)
        del combine_79

        # pd_op.reshape: (-1x49x3x24x32xf32) <- (-1x49x2304xf32, 5xi64)
        reshape_174 = paddle._C_ops.reshape(add_84, stack_76)
        del add_84, stack_76

        # pd_op.transpose: (3x-1x24x49x32xf32) <- (-1x49x3x24x32xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_174, [2, 0, 3, 1, 4])
        del reshape_174

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_2, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (-1x24x49x32xf32) <- (3x-1x24x49x32xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(
            transpose_74, [0], full_int_array_5, full_int_array_6, [1], [0]
        )
        del full_int_array_6, transpose_74

        # pd_op.scale: (-1x24x49x32xf32) <- (-1x24x49x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_85, full_7, float("0"), True)
        del full_7, slice_85

        # pd_op.transpose: (-1x24x32x49xf32) <- (-1x24x49x32xf32)
        transpose_75 = paddle._C_ops.transpose(slice_86, [0, 1, 3, 2])
        del slice_86

        # pd_op.matmul: (-1x24x49x49xf32) <- (-1x24x49x32xf32, -1x24x32x49xf32)
        matmul_70 = paddle._C_ops.matmul(scale_11, transpose_75, False, False)
        del scale_11, transpose_75

        # pd_op.reshape: (2401xi64) <- (49x49xi64, 1xi64)
        reshape_175 = paddle._C_ops.reshape(data_24, full_int_array_7)
        del data_24, full_int_array_7

        # pd_op.index_select: (2401x24xf32) <- (169x24xf32, 2401xi64)
        index_select_11 = paddle._C_ops.index_select(data_11, reshape_175, 0)
        del data_11, reshape_175

        # pd_op.reshape: (49x49x24xf32) <- (2401x24xf32, 3xi64)
        reshape_176 = paddle._C_ops.reshape(index_select_11, full_int_array_8)
        del full_int_array_8, index_select_11

        # pd_op.transpose: (24x49x49xf32) <- (49x49x24xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_176, [2, 0, 1])
        del reshape_176

        # pd_op.unsqueeze: (1x24x49x49xf32) <- (24x49x49xf32, 1xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(transpose_76, full_int_array_1)
        del transpose_76

        # pd_op.add: (-1x24x49x49xf32) <- (-1x24x49x49xf32, 1x24x49x49xf32)
        add_85 = paddle._C_ops.add(matmul_70, unsqueeze_33)
        del matmul_70, unsqueeze_33

        # pd_op.full: (xi64) <- ()
        full_45 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_5 = paddle._C_ops.floor_divide(slice_84, full_45)
        del full_45

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_80 = [floor_divide_5, full_40, full_41, full_4, full_4]
        del floor_divide_5, full_40

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_77 = paddle._C_ops.stack(combine_80, 0)
        del combine_80

        # pd_op.reshape: (-1x1x24x49x49xf32) <- (-1x24x49x49xf32, 5xi64)
        reshape_177 = paddle._C_ops.reshape(add_85, stack_77)
        del add_85, stack_77

        # pd_op.unsqueeze: (1x1x49x49xf32) <- (1x49x49xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(where_11, full_int_array_2)
        del full_int_array_2, where_11

        # pd_op.unsqueeze: (1x1x1x49x49xf32) <- (1x1x49x49xf32, 1xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(unsqueeze_34, full_int_array_1)
        del full_int_array_1, unsqueeze_34

        # pd_op.add: (-1x1x24x49x49xf32) <- (-1x1x24x49x49xf32, 1x1x1x49x49xf32)
        add_86 = paddle._C_ops.add(reshape_177, unsqueeze_35)
        del reshape_177, unsqueeze_35

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_81 = [slice_84, full_41, full_4, full_4]
        del full_41

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_78 = paddle._C_ops.stack(combine_81, 0)
        del combine_81

        # pd_op.reshape: (-1x24x49x49xf32) <- (-1x1x24x49x49xf32, 4xi64)
        reshape_178 = paddle._C_ops.reshape(add_86, stack_78)
        del add_86, stack_78

        # pd_op.softmax: (-1x24x49x49xf32) <- (-1x24x49x49xf32)
        softmax_11 = paddle._C_ops.softmax(reshape_178, -1)
        del reshape_178

        # pd_op.matmul: (-1x24x49x32xf32) <- (-1x24x49x49xf32, -1x24x49x32xf32)
        matmul_71 = paddle._C_ops.matmul(softmax_11, slice_87, False, False)
        del slice_87, softmax_11

        # pd_op.transpose: (-1x49x24x32xf32) <- (-1x24x49x32xf32)
        transpose_77 = paddle._C_ops.transpose(matmul_71, [0, 2, 1, 3])
        del matmul_71

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_82 = [slice_84, full_4, full_28]
        del slice_84

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_79 = paddle._C_ops.stack(combine_82, 0)
        del combine_82

        # pd_op.reshape: (-1x49x768xf32) <- (-1x49x24x32xf32, 3xi64)
        reshape_179 = paddle._C_ops.reshape(transpose_77, stack_79)
        del stack_79, transpose_77

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(reshape_179, parameter_11, False, False)
        del parameter_11, reshape_179

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        add_87 = paddle._C_ops.add(matmul_72, parameter_10)
        del matmul_72, parameter_10

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x49x768xf32, 4xi64)
        reshape_180 = paddle._C_ops.reshape(add_87, full_int_array_43)
        del add_87

        # pd_op.reshape: (-1x1x1x7x7x768xf32) <- (-1x7x7x768xf32, 6xi64)
        reshape_181 = paddle._C_ops.reshape(reshape_180, full_int_array_45)
        del full_int_array_45, reshape_180

        # pd_op.transpose: (-1x1x7x1x7x768xf32) <- (-1x1x1x7x7x768xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_181, [0, 1, 3, 2, 4, 5])
        del reshape_181

        # pd_op.reshape: (-1x7x7x768xf32) <- (-1x1x7x1x7x768xf32, 4xi64)
        reshape_182 = paddle._C_ops.reshape(transpose_78, full_int_array_43)
        del full_int_array_43, transpose_78

        # pd_op.roll: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 2xi64)
        roll_11 = paddle._C_ops.roll(reshape_182, full_int_array_29, [1, 2])
        del full_int_array_29, reshape_182

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_83 = [slice_81, full_4, full_28]
        del full_28, full_4, slice_81

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_80 = paddle._C_ops.stack(combine_83, 0)
        del combine_83

        # pd_op.reshape: (-1x49x768xf32) <- (-1x7x7x768xf32, 3xi64)
        reshape_183 = paddle._C_ops.reshape(roll_11, stack_80)
        del roll_11, stack_80

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, -1x49x768xf32)
        add_88 = paddle._C_ops.add(add_83, reshape_183)
        del add_83, reshape_183

        # pd_op.layer_norm: (-1x49x768xf32, -1x49xf32, -1x49xf32) <- (-1x49x768xf32, 768xf32, 768xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_88, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (-1x49x3072xf32) <- (-1x49x768xf32, 768x3072xf32)
        matmul_73 = paddle._C_ops.matmul(layer_norm_81, parameter_7, False, False)
        del layer_norm_81, parameter_7

        # pd_op.add: (-1x49x3072xf32) <- (-1x49x3072xf32, 3072xf32)
        add_89 = paddle._C_ops.add(matmul_73, parameter_6)
        del matmul_73, parameter_6

        # pd_op.gelu: (-1x49x3072xf32) <- (-1x49x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_89, False)
        del add_89

        # pd_op.matmul: (-1x49x768xf32) <- (-1x49x3072xf32, 3072x768xf32)
        matmul_74 = paddle._C_ops.matmul(gelu_11, parameter_5, False, False)
        del gelu_11, parameter_5

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_74, parameter_4)
        del matmul_74, parameter_4

        # pd_op.add: (-1x49x768xf32) <- (-1x49x768xf32, -1x49x768xf32)
        add_91 = paddle._C_ops.add(add_88, add_90)
        del add_88, add_90

        # pd_op.layer_norm: (-1x49x768xf32, -1x49xf32, -1x49xf32) <- (-1x49x768xf32, 768xf32, 768xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_91, parameter_3, parameter_2, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_91, parameter_2, parameter_3

        # pd_op.transpose: (-1x768x49xf32) <- (-1x49x768xf32)
        transpose_79 = paddle._C_ops.transpose(layer_norm_84, [0, 2, 1])
        del layer_norm_84

        # pd_op.unsqueeze: (-1x768x1x49xf32) <- (-1x768x49xf32, 1xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(transpose_79, full_int_array_5)
        del transpose_79

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x1x49xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            unsqueeze_36,
            full_int_array_14,
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
        del full_int_array_14, unsqueeze_36

        # pd_op.squeeze: (-1x768x1xf32) <- (-1x768x1x1xf32, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(pool2d_0, full_int_array_5)
        del full_int_array_5, pool2d_0

        # pd_op.flatten: (-1x768xf32) <- (-1x768x1xf32)
        flatten_1 = paddle._C_ops.flatten(squeeze_0, 1, 2)
        del squeeze_0

        # pd_op.matmul: (-1x102xf32) <- (-1x768xf32, 768x102xf32)
        matmul_75 = paddle._C_ops.matmul(flatten_1, parameter_1, False, False)
        del flatten_1, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_75, parameter_0)
        del (
            matmul_75,
            parameter_0,
            set_value__17,
            set_value__26,
            set_value__35,
            set_value__44,
            set_value__53,
            set_value__8,
        )

        return add_0
