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
        data_25,
        data_26,
    ):
        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_26, parameter_155, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_26, parameter_155

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_154, full_int_array_0)
        del parameter_154

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)
        del conv2d_0, reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(add_1, full_int_array_1, True)

        # pd_op.subtract: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        subtract_0 = paddle._C_ops.subtract(add_1, mean_0)
        del add_1, mean_0

        # pd_op.pow: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(pow_0, full_int_array_1, True)
        del pow_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(mean_1, full_0, float("1e-06"), True)
        del mean_1

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_0)
        del scale_0

        # pd_op.divide: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        divide_0 = paddle._C_ops.divide(subtract_0, sqrt_0)
        del sqrt_0, subtract_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 2]

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_2)
        del data_0

        # pd_op.multiply: (-1x96x56x56xf32) <- (96x1x1xf32, -1x96x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze_0, divide_0)
        del divide_0, unsqueeze_0

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_2)
        del data_1

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x1xf32)
        add_2 = paddle._C_ops.add(multiply_0, unsqueeze_1)
        del multiply_0, unsqueeze_1

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            add_2, parameter_153, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_153

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_152, full_int_array_0)
        del parameter_152

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_3 = paddle._C_ops.add(depthwise_conv2d_0, reshape_1)
        del depthwise_conv2d_0, reshape_1

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        transpose_0 = paddle._C_ops.transpose(add_3, [0, 2, 3, 1])
        del add_3

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_151, parameter_150, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_150, parameter_151, transpose_0

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_149, False, False)
        del layer_norm_0, parameter_149

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_148)
        del matmul_0, parameter_148

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)
        del add_4

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_147, False, False)
        del gelu_0, parameter_147

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        add_5 = paddle._C_ops.add(matmul_1, parameter_146)
        del matmul_1, parameter_146

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        multiply_1 = paddle._C_ops.multiply(data_2, add_5)
        del add_5, data_2

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        transpose_1 = paddle._C_ops.transpose(multiply_1, [0, 3, 1, 2])
        del multiply_1

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        add_6 = paddle._C_ops.add(add_2, transpose_1)
        del add_2, transpose_1

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            add_6, parameter_145, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_145

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_144, full_int_array_0)
        del parameter_144

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_1, reshape_2)
        del depthwise_conv2d_1, reshape_2

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        transpose_2 = paddle._C_ops.transpose(add_7, [0, 2, 3, 1])
        del add_7

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_143, parameter_142, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_142, parameter_143, transpose_2

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_3, parameter_141, False, False)
        del layer_norm_3, parameter_141

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_2, parameter_140)
        del matmul_2, parameter_140

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_8, False)
        del add_8

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_139, False, False)
        del gelu_1, parameter_139

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        add_9 = paddle._C_ops.add(matmul_3, parameter_138)
        del matmul_3, parameter_138

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        multiply_2 = paddle._C_ops.multiply(data_3, add_9)
        del add_9, data_3

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        transpose_3 = paddle._C_ops.transpose(multiply_2, [0, 3, 1, 2])
        del multiply_2

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        add_10 = paddle._C_ops.add(add_6, transpose_3)
        del add_6, transpose_3

        # pd_op.depthwise_conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            add_10, parameter_137, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_137

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_136, full_int_array_0)
        del parameter_136

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 1x96x1x1xf32)
        add_11 = paddle._C_ops.add(depthwise_conv2d_2, reshape_3)
        del depthwise_conv2d_2, reshape_3

        # pd_op.transpose: (-1x56x56x96xf32) <- (-1x96x56x56xf32)
        transpose_4 = paddle._C_ops.transpose(add_11, [0, 2, 3, 1])
        del add_11

        # pd_op.layer_norm: (-1x56x56x96xf32, -1x56x56xf32, -1x56x56xf32) <- (-1x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_135, parameter_134, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135, transpose_4

        # pd_op.matmul: (-1x56x56x384xf32) <- (-1x56x56x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_133, False, False)
        del layer_norm_6, parameter_133

        # pd_op.add: (-1x56x56x384xf32) <- (-1x56x56x384xf32, 384xf32)
        add_12 = paddle._C_ops.add(matmul_4, parameter_132)
        del matmul_4, parameter_132

        # pd_op.gelu: (-1x56x56x384xf32) <- (-1x56x56x384xf32)
        gelu_2 = paddle._C_ops.gelu(add_12, False)
        del add_12

        # pd_op.matmul: (-1x56x56x96xf32) <- (-1x56x56x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_2, parameter_131, False, False)
        del gelu_2, parameter_131

        # pd_op.add: (-1x56x56x96xf32) <- (-1x56x56x96xf32, 96xf32)
        add_13 = paddle._C_ops.add(matmul_5, parameter_130)
        del matmul_5, parameter_130

        # pd_op.multiply: (-1x56x56x96xf32) <- (96xf32, -1x56x56x96xf32)
        multiply_3 = paddle._C_ops.multiply(data_4, add_13)
        del add_13, data_4

        # pd_op.transpose: (-1x96x56x56xf32) <- (-1x56x56x96xf32)
        transpose_5 = paddle._C_ops.transpose(multiply_3, [0, 3, 1, 2])
        del multiply_3

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x96x56x56xf32)
        add_14 = paddle._C_ops.add(add_10, transpose_5)
        del add_10, transpose_5

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(add_14, full_int_array_1, True)

        # pd_op.subtract: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        subtract_1 = paddle._C_ops.subtract(add_14, mean_2)
        del add_14, mean_2

        # pd_op.pow: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        pow_1 = paddle._C_ops.pow(subtract_1, float("2"))

        # pd_op.mean: (-1x1x56x56xf32) <- (-1x96x56x56xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(pow_1, full_int_array_1, True)
        del pow_1

        # pd_op.scale: (-1x1x56x56xf32) <- (-1x1x56x56xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(mean_3, full_0, float("1e-06"), True)
        del mean_3

        # pd_op.sqrt: (-1x1x56x56xf32) <- (-1x1x56x56xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_1)
        del scale_1

        # pd_op.divide: (-1x96x56x56xf32) <- (-1x96x56x56xf32, -1x1x56x56xf32)
        divide_1 = paddle._C_ops.divide(subtract_1, sqrt_1)
        del sqrt_1, subtract_1

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_5, full_int_array_2)
        del data_5

        # pd_op.multiply: (-1x96x56x56xf32) <- (96x1x1xf32, -1x96x56x56xf32)
        multiply_4 = paddle._C_ops.multiply(unsqueeze_2, divide_1)
        del divide_1, unsqueeze_2

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(data_6, full_int_array_2)
        del data_6

        # pd_op.add: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x1x1xf32)
        add_15 = paddle._C_ops.add(multiply_4, unsqueeze_3)
        del multiply_4, unsqueeze_3

        # pd_op.conv2d: (-1x192x28x28xf32) <- (-1x96x56x56xf32, 192x96x2x2xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            add_15, parameter_129, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_15, parameter_129

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_128, full_int_array_0)
        del parameter_128

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_1, reshape_4)
        del conv2d_1, reshape_4

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            add_16, parameter_127, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_127

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_126, full_int_array_0)
        del parameter_126

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_17 = paddle._C_ops.add(depthwise_conv2d_3, reshape_5)
        del depthwise_conv2d_3, reshape_5

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        transpose_6 = paddle._C_ops.transpose(add_17, [0, 2, 3, 1])
        del add_17

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_125, parameter_124, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125, transpose_6

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_123, False, False)
        del layer_norm_9, parameter_123

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(matmul_6, parameter_122)
        del matmul_6, parameter_122

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_18, False)
        del add_18

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_3, parameter_121, False, False)
        del gelu_3, parameter_121

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        add_19 = paddle._C_ops.add(matmul_7, parameter_120)
        del matmul_7, parameter_120

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        multiply_5 = paddle._C_ops.multiply(data_7, add_19)
        del add_19, data_7

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        transpose_7 = paddle._C_ops.transpose(multiply_5, [0, 3, 1, 2])
        del multiply_5

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        add_20 = paddle._C_ops.add(add_16, transpose_7)
        del add_16, transpose_7

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            add_20, parameter_119, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_118, full_int_array_0)
        del parameter_118

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_21 = paddle._C_ops.add(depthwise_conv2d_4, reshape_6)
        del depthwise_conv2d_4, reshape_6

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        transpose_8 = paddle._C_ops.transpose(add_21, [0, 2, 3, 1])
        del add_21

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_8, parameter_117, parameter_116, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117, transpose_8

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_12, parameter_115, False, False)
        del layer_norm_12, parameter_115

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        add_22 = paddle._C_ops.add(matmul_8, parameter_114)
        del matmul_8, parameter_114

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        gelu_4 = paddle._C_ops.gelu(add_22, False)
        del add_22

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_4, parameter_113, False, False)
        del gelu_4, parameter_113

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        add_23 = paddle._C_ops.add(matmul_9, parameter_112)
        del matmul_9, parameter_112

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        multiply_6 = paddle._C_ops.multiply(data_8, add_23)
        del add_23, data_8

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        transpose_9 = paddle._C_ops.transpose(multiply_6, [0, 3, 1, 2])
        del multiply_6

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        add_24 = paddle._C_ops.add(add_20, transpose_9)
        del add_20, transpose_9

        # pd_op.depthwise_conv2d: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            add_24, parameter_111, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_111

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_110, full_int_array_0)
        del parameter_110

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 1x192x1x1xf32)
        add_25 = paddle._C_ops.add(depthwise_conv2d_5, reshape_7)
        del depthwise_conv2d_5, reshape_7

        # pd_op.transpose: (-1x28x28x192xf32) <- (-1x192x28x28xf32)
        transpose_10 = paddle._C_ops.transpose(add_25, [0, 2, 3, 1])
        del add_25

        # pd_op.layer_norm: (-1x28x28x192xf32, -1x28x28xf32, -1x28x28xf32) <- (-1x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_10, parameter_109, parameter_108, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_108, parameter_109, transpose_10

        # pd_op.matmul: (-1x28x28x768xf32) <- (-1x28x28x192xf32, 192x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_15, parameter_107, False, False)
        del layer_norm_15, parameter_107

        # pd_op.add: (-1x28x28x768xf32) <- (-1x28x28x768xf32, 768xf32)
        add_26 = paddle._C_ops.add(matmul_10, parameter_106)
        del matmul_10, parameter_106

        # pd_op.gelu: (-1x28x28x768xf32) <- (-1x28x28x768xf32)
        gelu_5 = paddle._C_ops.gelu(add_26, False)
        del add_26

        # pd_op.matmul: (-1x28x28x192xf32) <- (-1x28x28x768xf32, 768x192xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_5, parameter_105, False, False)
        del gelu_5, parameter_105

        # pd_op.add: (-1x28x28x192xf32) <- (-1x28x28x192xf32, 192xf32)
        add_27 = paddle._C_ops.add(matmul_11, parameter_104)
        del matmul_11, parameter_104

        # pd_op.multiply: (-1x28x28x192xf32) <- (192xf32, -1x28x28x192xf32)
        multiply_7 = paddle._C_ops.multiply(data_9, add_27)
        del add_27, data_9

        # pd_op.transpose: (-1x192x28x28xf32) <- (-1x28x28x192xf32)
        transpose_11 = paddle._C_ops.transpose(multiply_7, [0, 3, 1, 2])
        del multiply_7

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x192x28x28xf32)
        add_28 = paddle._C_ops.add(add_24, transpose_11)
        del add_24, transpose_11

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x192x28x28xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(add_28, full_int_array_1, True)

        # pd_op.subtract: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x28x28xf32)
        subtract_2 = paddle._C_ops.subtract(add_28, mean_4)
        del add_28, mean_4

        # pd_op.pow: (-1x192x28x28xf32) <- (-1x192x28x28xf32)
        pow_2 = paddle._C_ops.pow(subtract_2, float("2"))

        # pd_op.mean: (-1x1x28x28xf32) <- (-1x192x28x28xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(pow_2, full_int_array_1, True)
        del pow_2

        # pd_op.scale: (-1x1x28x28xf32) <- (-1x1x28x28xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(mean_5, full_0, float("1e-06"), True)
        del mean_5

        # pd_op.sqrt: (-1x1x28x28xf32) <- (-1x1x28x28xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_2)
        del scale_2

        # pd_op.divide: (-1x192x28x28xf32) <- (-1x192x28x28xf32, -1x1x28x28xf32)
        divide_2 = paddle._C_ops.divide(subtract_2, sqrt_2)
        del sqrt_2, subtract_2

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(data_10, full_int_array_2)
        del data_10

        # pd_op.multiply: (-1x192x28x28xf32) <- (192x1x1xf32, -1x192x28x28xf32)
        multiply_8 = paddle._C_ops.multiply(unsqueeze_4, divide_2)
        del divide_2, unsqueeze_4

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(data_11, full_int_array_2)
        del data_11

        # pd_op.add: (-1x192x28x28xf32) <- (-1x192x28x28xf32, 192x1x1xf32)
        add_29 = paddle._C_ops.add(multiply_8, unsqueeze_5)
        del multiply_8, unsqueeze_5

        # pd_op.conv2d: (-1x384x14x14xf32) <- (-1x192x28x28xf32, 384x192x2x2xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_29, parameter_103, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_29, parameter_103

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_102, full_int_array_0)
        del parameter_102

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_2, reshape_8)
        del conv2d_2, reshape_8

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            add_30, parameter_101, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_100, full_int_array_0)
        del parameter_100

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_31 = paddle._C_ops.add(depthwise_conv2d_6, reshape_9)
        del depthwise_conv2d_6, reshape_9

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_12 = paddle._C_ops.transpose(add_31, [0, 2, 3, 1])
        del add_31

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_12, parameter_99, parameter_98, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99, transpose_12

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_97, False, False)
        del layer_norm_18, parameter_97

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_32 = paddle._C_ops.add(matmul_12, parameter_96)
        del matmul_12, parameter_96

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_32, False)
        del add_32

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_13 = paddle._C_ops.matmul(gelu_6, parameter_95, False, False)
        del gelu_6, parameter_95

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_33 = paddle._C_ops.add(matmul_13, parameter_94)
        del matmul_13, parameter_94

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_9 = paddle._C_ops.multiply(data_12, add_33)
        del add_33, data_12

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_13 = paddle._C_ops.transpose(multiply_9, [0, 3, 1, 2])
        del multiply_9

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_34 = paddle._C_ops.add(add_30, transpose_13)
        del add_30, transpose_13

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            add_34, parameter_93, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_93

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_92, full_int_array_0)
        del parameter_92

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_35 = paddle._C_ops.add(depthwise_conv2d_7, reshape_10)
        del depthwise_conv2d_7, reshape_10

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_14 = paddle._C_ops.transpose(add_35, [0, 2, 3, 1])
        del add_35

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_14, parameter_91, parameter_90, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_90, parameter_91, transpose_14

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_21, parameter_89, False, False)
        del layer_norm_21, parameter_89

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_36 = paddle._C_ops.add(matmul_14, parameter_88)
        del matmul_14, parameter_88

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_36, False)
        del add_36

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_7, parameter_87, False, False)
        del gelu_7, parameter_87

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_37 = paddle._C_ops.add(matmul_15, parameter_86)
        del matmul_15, parameter_86

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_10 = paddle._C_ops.multiply(data_13, add_37)
        del add_37, data_13

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_15 = paddle._C_ops.transpose(multiply_10, [0, 3, 1, 2])
        del multiply_10

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_38 = paddle._C_ops.add(add_34, transpose_15)
        del add_34, transpose_15

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            add_38, parameter_85, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_85

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_84, full_int_array_0)
        del parameter_84

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_39 = paddle._C_ops.add(depthwise_conv2d_8, reshape_11)
        del depthwise_conv2d_8, reshape_11

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_16 = paddle._C_ops.transpose(add_39, [0, 2, 3, 1])
        del add_39

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_16, parameter_83, parameter_82, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83, transpose_16

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_24, parameter_81, False, False)
        del layer_norm_24, parameter_81

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_40 = paddle._C_ops.add(matmul_16, parameter_80)
        del matmul_16, parameter_80

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_40, False)
        del add_40

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_8, parameter_79, False, False)
        del gelu_8, parameter_79

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_41 = paddle._C_ops.add(matmul_17, parameter_78)
        del matmul_17, parameter_78

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_11 = paddle._C_ops.multiply(data_14, add_41)
        del add_41, data_14

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_17 = paddle._C_ops.transpose(multiply_11, [0, 3, 1, 2])
        del multiply_11

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_42 = paddle._C_ops.add(add_38, transpose_17)
        del add_38, transpose_17

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            add_42, parameter_77, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_77

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_76, full_int_array_0)
        del parameter_76

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_43 = paddle._C_ops.add(depthwise_conv2d_9, reshape_12)
        del depthwise_conv2d_9, reshape_12

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_18 = paddle._C_ops.transpose(add_43, [0, 2, 3, 1])
        del add_43

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_75, parameter_74, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75, transpose_18

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_27, parameter_73, False, False)
        del layer_norm_27, parameter_73

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_44 = paddle._C_ops.add(matmul_18, parameter_72)
        del matmul_18, parameter_72

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_44, False)
        del add_44

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_19 = paddle._C_ops.matmul(gelu_9, parameter_71, False, False)
        del gelu_9, parameter_71

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_45 = paddle._C_ops.add(matmul_19, parameter_70)
        del matmul_19, parameter_70

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_12 = paddle._C_ops.multiply(data_15, add_45)
        del add_45, data_15

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_19 = paddle._C_ops.transpose(multiply_12, [0, 3, 1, 2])
        del multiply_12

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_46 = paddle._C_ops.add(add_42, transpose_19)
        del add_42, transpose_19

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            add_46, parameter_69, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_69

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_68, full_int_array_0)
        del parameter_68

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_47 = paddle._C_ops.add(depthwise_conv2d_10, reshape_13)
        del depthwise_conv2d_10, reshape_13

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_20 = paddle._C_ops.transpose(add_47, [0, 2, 3, 1])
        del add_47

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_20, parameter_67, parameter_66, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67, transpose_20

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_30, parameter_65, False, False)
        del layer_norm_30, parameter_65

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_48 = paddle._C_ops.add(matmul_20, parameter_64)
        del matmul_20, parameter_64

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_10 = paddle._C_ops.gelu(add_48, False)
        del add_48

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_10, parameter_63, False, False)
        del gelu_10, parameter_63

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_49 = paddle._C_ops.add(matmul_21, parameter_62)
        del matmul_21, parameter_62

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_13 = paddle._C_ops.multiply(data_16, add_49)
        del add_49, data_16

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_21 = paddle._C_ops.transpose(multiply_13, [0, 3, 1, 2])
        del multiply_13

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_50 = paddle._C_ops.add(add_46, transpose_21)
        del add_46, transpose_21

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            add_50, parameter_61, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_61

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_60, full_int_array_0)
        del parameter_60

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_51 = paddle._C_ops.add(depthwise_conv2d_11, reshape_14)
        del depthwise_conv2d_11, reshape_14

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_22 = paddle._C_ops.transpose(add_51, [0, 2, 3, 1])
        del add_51

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_22, parameter_59, parameter_58, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59, transpose_22

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_33, parameter_57, False, False)
        del layer_norm_33, parameter_57

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_52 = paddle._C_ops.add(matmul_22, parameter_56)
        del matmul_22, parameter_56

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_11 = paddle._C_ops.gelu(add_52, False)
        del add_52

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_11, parameter_55, False, False)
        del gelu_11, parameter_55

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_53 = paddle._C_ops.add(matmul_23, parameter_54)
        del matmul_23, parameter_54

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_14 = paddle._C_ops.multiply(data_17, add_53)
        del add_53, data_17

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_23 = paddle._C_ops.transpose(multiply_14, [0, 3, 1, 2])
        del multiply_14

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_54 = paddle._C_ops.add(add_50, transpose_23)
        del add_50, transpose_23

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            add_54, parameter_53, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_52, full_int_array_0)
        del parameter_52

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_55 = paddle._C_ops.add(depthwise_conv2d_12, reshape_15)
        del depthwise_conv2d_12, reshape_15

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_24 = paddle._C_ops.transpose(add_55, [0, 2, 3, 1])
        del add_55

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_24, parameter_51, parameter_50, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51, transpose_24

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_36, parameter_49, False, False)
        del layer_norm_36, parameter_49

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_56 = paddle._C_ops.add(matmul_24, parameter_48)
        del matmul_24, parameter_48

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_12 = paddle._C_ops.gelu(add_56, False)
        del add_56

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_25 = paddle._C_ops.matmul(gelu_12, parameter_47, False, False)
        del gelu_12, parameter_47

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_57 = paddle._C_ops.add(matmul_25, parameter_46)
        del matmul_25, parameter_46

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_15 = paddle._C_ops.multiply(data_18, add_57)
        del add_57, data_18

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_25 = paddle._C_ops.transpose(multiply_15, [0, 3, 1, 2])
        del multiply_15

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_58 = paddle._C_ops.add(add_54, transpose_25)
        del add_54, transpose_25

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            add_58, parameter_45, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_45

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_44, full_int_array_0)
        del parameter_44

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_59 = paddle._C_ops.add(depthwise_conv2d_13, reshape_16)
        del depthwise_conv2d_13, reshape_16

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_26 = paddle._C_ops.transpose(add_59, [0, 2, 3, 1])
        del add_59

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_26, parameter_43, parameter_42, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43, transpose_26

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_39, parameter_41, False, False)
        del layer_norm_39, parameter_41

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_60 = paddle._C_ops.add(matmul_26, parameter_40)
        del matmul_26, parameter_40

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_13 = paddle._C_ops.gelu(add_60, False)
        del add_60

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_13, parameter_39, False, False)
        del gelu_13, parameter_39

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_61 = paddle._C_ops.add(matmul_27, parameter_38)
        del matmul_27, parameter_38

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_16 = paddle._C_ops.multiply(data_19, add_61)
        del add_61, data_19

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_27 = paddle._C_ops.transpose(multiply_16, [0, 3, 1, 2])
        del multiply_16

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_62 = paddle._C_ops.add(add_58, transpose_27)
        del add_58, transpose_27

        # pd_op.depthwise_conv2d: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            add_62, parameter_37, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_37

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_36, full_int_array_0)
        del parameter_36

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 1x384x1x1xf32)
        add_63 = paddle._C_ops.add(depthwise_conv2d_14, reshape_17)
        del depthwise_conv2d_14, reshape_17

        # pd_op.transpose: (-1x14x14x384xf32) <- (-1x384x14x14xf32)
        transpose_28 = paddle._C_ops.transpose(add_63, [0, 2, 3, 1])
        del add_63

        # pd_op.layer_norm: (-1x14x14x384xf32, -1x14x14xf32, -1x14x14xf32) <- (-1x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_28, parameter_35, parameter_34, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35, transpose_28

        # pd_op.matmul: (-1x14x14x1536xf32) <- (-1x14x14x384xf32, 384x1536xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_42, parameter_33, False, False)
        del layer_norm_42, parameter_33

        # pd_op.add: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32, 1536xf32)
        add_64 = paddle._C_ops.add(matmul_28, parameter_32)
        del matmul_28, parameter_32

        # pd_op.gelu: (-1x14x14x1536xf32) <- (-1x14x14x1536xf32)
        gelu_14 = paddle._C_ops.gelu(add_64, False)
        del add_64

        # pd_op.matmul: (-1x14x14x384xf32) <- (-1x14x14x1536xf32, 1536x384xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_14, parameter_31, False, False)
        del gelu_14, parameter_31

        # pd_op.add: (-1x14x14x384xf32) <- (-1x14x14x384xf32, 384xf32)
        add_65 = paddle._C_ops.add(matmul_29, parameter_30)
        del matmul_29, parameter_30

        # pd_op.multiply: (-1x14x14x384xf32) <- (384xf32, -1x14x14x384xf32)
        multiply_17 = paddle._C_ops.multiply(data_20, add_65)
        del add_65, data_20

        # pd_op.transpose: (-1x384x14x14xf32) <- (-1x14x14x384xf32)
        transpose_29 = paddle._C_ops.transpose(multiply_17, [0, 3, 1, 2])
        del multiply_17

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x384x14x14xf32)
        add_66 = paddle._C_ops.add(add_62, transpose_29)
        del add_62, transpose_29

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x384x14x14xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(add_66, full_int_array_1, True)

        # pd_op.subtract: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x14x14xf32)
        subtract_3 = paddle._C_ops.subtract(add_66, mean_6)
        del add_66, mean_6

        # pd_op.pow: (-1x384x14x14xf32) <- (-1x384x14x14xf32)
        pow_3 = paddle._C_ops.pow(subtract_3, float("2"))

        # pd_op.mean: (-1x1x14x14xf32) <- (-1x384x14x14xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(pow_3, full_int_array_1, True)
        del full_int_array_1, pow_3

        # pd_op.scale: (-1x1x14x14xf32) <- (-1x1x14x14xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(mean_7, full_0, float("1e-06"), True)
        del full_0, mean_7

        # pd_op.sqrt: (-1x1x14x14xf32) <- (-1x1x14x14xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_3)
        del scale_3

        # pd_op.divide: (-1x384x14x14xf32) <- (-1x384x14x14xf32, -1x1x14x14xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, sqrt_3)
        del sqrt_3, subtract_3

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(data_21, full_int_array_2)
        del data_21

        # pd_op.multiply: (-1x384x14x14xf32) <- (384x1x1xf32, -1x384x14x14xf32)
        multiply_18 = paddle._C_ops.multiply(unsqueeze_6, divide_3)
        del divide_3, unsqueeze_6

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(data_22, full_int_array_2)
        del data_22, full_int_array_2

        # pd_op.add: (-1x384x14x14xf32) <- (-1x384x14x14xf32, 384x1x1xf32)
        add_67 = paddle._C_ops.add(multiply_18, unsqueeze_7)
        del multiply_18, unsqueeze_7

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x384x14x14xf32, 768x384x2x2xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_67, parameter_29, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_67, parameter_29

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_28, full_int_array_0)
        del parameter_28

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_68 = paddle._C_ops.add(conv2d_3, reshape_18)
        del conv2d_3, reshape_18

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            add_68, parameter_27, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_27

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_26, full_int_array_0)
        del parameter_26

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_69 = paddle._C_ops.add(depthwise_conv2d_15, reshape_19)
        del depthwise_conv2d_15, reshape_19

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        transpose_30 = paddle._C_ops.transpose(add_69, [0, 2, 3, 1])
        del add_69

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_30, parameter_25, parameter_24, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25, transpose_30

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_45, parameter_23, False, False)
        del layer_norm_45, parameter_23

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        add_70 = paddle._C_ops.add(matmul_30, parameter_22)
        del matmul_30, parameter_22

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        gelu_15 = paddle._C_ops.gelu(add_70, False)
        del add_70

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_15, parameter_21, False, False)
        del gelu_15, parameter_21

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        add_71 = paddle._C_ops.add(matmul_31, parameter_20)
        del matmul_31, parameter_20

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        multiply_19 = paddle._C_ops.multiply(data_23, add_71)
        del add_71, data_23

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        transpose_31 = paddle._C_ops.transpose(multiply_19, [0, 3, 1, 2])
        del multiply_19

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        add_72 = paddle._C_ops.add(add_68, transpose_31)
        del add_68, transpose_31

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            add_72, parameter_19, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_19

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_18, full_int_array_0)
        del parameter_18

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_73 = paddle._C_ops.add(depthwise_conv2d_16, reshape_20)
        del depthwise_conv2d_16, reshape_20

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        transpose_32 = paddle._C_ops.transpose(add_73, [0, 2, 3, 1])
        del add_73

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_32, parameter_17, parameter_16, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17, transpose_32

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_48, parameter_15, False, False)
        del layer_norm_48, parameter_15

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        add_74 = paddle._C_ops.add(matmul_32, parameter_14)
        del matmul_32, parameter_14

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        gelu_16 = paddle._C_ops.gelu(add_74, False)
        del add_74

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        matmul_33 = paddle._C_ops.matmul(gelu_16, parameter_13, False, False)
        del gelu_16, parameter_13

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        add_75 = paddle._C_ops.add(matmul_33, parameter_12)
        del matmul_33, parameter_12

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        multiply_20 = paddle._C_ops.multiply(data_24, add_75)
        del add_75, data_24

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        transpose_33 = paddle._C_ops.transpose(multiply_20, [0, 3, 1, 2])
        del multiply_20

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        add_76 = paddle._C_ops.add(add_72, transpose_33)
        del add_72, transpose_33

        # pd_op.depthwise_conv2d: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            add_76, parameter_11, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_11

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_10, full_int_array_0)
        del full_int_array_0, parameter_10

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, 1x768x1x1xf32)
        add_77 = paddle._C_ops.add(depthwise_conv2d_17, reshape_21)
        del depthwise_conv2d_17, reshape_21

        # pd_op.transpose: (-1x7x7x768xf32) <- (-1x768x7x7xf32)
        transpose_34 = paddle._C_ops.transpose(add_77, [0, 2, 3, 1])
        del add_77

        # pd_op.layer_norm: (-1x7x7x768xf32, -1x7x7xf32, -1x7x7xf32) <- (-1x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_34, parameter_9, parameter_8, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9, transpose_34

        # pd_op.matmul: (-1x7x7x3072xf32) <- (-1x7x7x768xf32, 768x3072xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_51, parameter_7, False, False)
        del layer_norm_51, parameter_7

        # pd_op.add: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32, 3072xf32)
        add_78 = paddle._C_ops.add(matmul_34, parameter_6)
        del matmul_34, parameter_6

        # pd_op.gelu: (-1x7x7x3072xf32) <- (-1x7x7x3072xf32)
        gelu_17 = paddle._C_ops.gelu(add_78, False)
        del add_78

        # pd_op.matmul: (-1x7x7x768xf32) <- (-1x7x7x3072xf32, 3072x768xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_17, parameter_5, False, False)
        del gelu_17, parameter_5

        # pd_op.add: (-1x7x7x768xf32) <- (-1x7x7x768xf32, 768xf32)
        add_79 = paddle._C_ops.add(matmul_35, parameter_4)
        del matmul_35, parameter_4

        # pd_op.multiply: (-1x7x7x768xf32) <- (768xf32, -1x7x7x768xf32)
        multiply_21 = paddle._C_ops.multiply(data_25, add_79)
        del add_79, data_25

        # pd_op.transpose: (-1x768x7x7xf32) <- (-1x7x7x768xf32)
        transpose_35 = paddle._C_ops.transpose(multiply_21, [0, 3, 1, 2])
        del multiply_21

        # pd_op.add: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x7x7xf32)
        add_80 = paddle._C_ops.add(add_76, transpose_35)
        del add_76, transpose_35

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [-2, -1]

        # pd_op.mean: (-1x768xf32) <- (-1x768x7x7xf32, 2xi64)
        mean_8 = paddle._C_ops.mean(add_80, full_int_array_3, False)
        del add_80, full_int_array_3

        # pd_op.layer_norm: (-1x768xf32, -1xf32, -1xf32) <- (-1x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                mean_8, parameter_3, parameter_2, float("1e-06"), 1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del mean_8, parameter_2, parameter_3

        # pd_op.matmul: (-1x102xf32) <- (-1x768xf32, 768x102xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_54, parameter_1, False, False)
        del layer_norm_54, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_36, parameter_0)
        del matmul_36, parameter_0

        return add_0
