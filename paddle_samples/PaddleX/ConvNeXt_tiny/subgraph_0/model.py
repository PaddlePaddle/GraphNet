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
        # pd_op.conv2d: (32x96x56x56xf32) <- (32x3x224x224xf32, 96x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_26, parameter_155, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_26, parameter_155

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_154, full_int_array_0)
        del parameter_154

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 1x96x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_1

        # pd_op.mean: (32x1x56x56xf32) <- (32x96x56x56xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(add_1, full_int_array_1, True)

        # pd_op.subtract: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x56x56xf32)
        subtract_0 = paddle._C_ops.subtract(add_1, mean_0)

        # pd_op.assign: (32x96x56x56xf32) <- (32x96x56x56xf32)
        assign_7 = subtract_0

        # pd_op.pow: (32x96x56x56xf32) <- (32x96x56x56xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))

        # pd_op.mean: (32x1x56x56xf32) <- (32x96x56x56xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(pow_0, full_int_array_1, True)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_0

        # pd_op.scale: (32x1x56x56xf32) <- (32x1x56x56xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(mean_1, full_0, float("1e-06"), True)
        del mean_1

        # pd_op.sqrt: (32x1x56x56xf32) <- (32x1x56x56xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_0)
        del scale_0

        # pd_op.divide: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x56x56xf32)
        divide_0 = paddle._C_ops.divide(subtract_0, sqrt_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 2]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_11 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_12 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_13 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_14 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_15 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_16 = full_int_array_2

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_17 = full_int_array_2

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_0, full_int_array_2)
        del data_0

        # pd_op.multiply: (32x96x56x56xf32) <- (96x1x1xf32, 32x96x56x56xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze_0, divide_0)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_2)
        del data_1

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 96x1x1xf32)
        add_2 = paddle._C_ops.add(multiply_0, unsqueeze_1)

        # pd_op.depthwise_conv2d: (32x96x56x56xf32) <- (32x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            add_2, parameter_153, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_153

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_152, full_int_array_0)
        del parameter_152

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 1x96x1x1xf32)
        add_3 = paddle._C_ops.add(depthwise_conv2d_0, reshape_1)

        # pd_op.transpose: (32x56x56x96xf32) <- (32x96x56x56xf32)
        transpose_0 = paddle._C_ops.transpose(add_3, [0, 2, 3, 1])
        del add_3

        # pd_op.layer_norm: (32x56x56x96xf32, 32x56x56xf32, 32x56x56xf32) <- (32x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_0, parameter_151, parameter_150, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_150, parameter_151

        # pd_op.matmul: (32x56x56x384xf32) <- (32x56x56x96xf32, 96x384xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_149, False, False)
        del parameter_149

        # pd_op.add: (32x56x56x384xf32) <- (32x56x56x384xf32, 384xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_148)
        del parameter_148

        # pd_op.gelu: (32x56x56x384xf32) <- (32x56x56x384xf32)
        gelu_0 = paddle._C_ops.gelu(add_4, False)

        # pd_op.matmul: (32x56x56x96xf32) <- (32x56x56x384xf32, 384x96xf32)
        matmul_1 = paddle._C_ops.matmul(gelu_0, parameter_147, False, False)
        del parameter_147

        # pd_op.add: (32x56x56x96xf32) <- (32x56x56x96xf32, 96xf32)
        add_5 = paddle._C_ops.add(matmul_1, parameter_146)
        del parameter_146

        # pd_op.multiply: (32x56x56x96xf32) <- (96xf32, 32x56x56x96xf32)
        multiply_1 = paddle._C_ops.multiply(data_12, add_5)
        del data_12

        # pd_op.transpose: (32x96x56x56xf32) <- (32x56x56x96xf32)
        transpose_1 = paddle._C_ops.transpose(multiply_1, [0, 3, 1, 2])
        del multiply_1

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x96x56x56xf32)
        add_6 = paddle._C_ops.add(add_2, transpose_1)

        # pd_op.depthwise_conv2d: (32x96x56x56xf32) <- (32x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            add_6, parameter_145, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_145

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_144, full_int_array_0)
        del parameter_144

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 1x96x1x1xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_1, reshape_2)

        # pd_op.transpose: (32x56x56x96xf32) <- (32x96x56x56xf32)
        transpose_2 = paddle._C_ops.transpose(add_7, [0, 2, 3, 1])
        del add_7

        # pd_op.layer_norm: (32x56x56x96xf32, 32x56x56xf32, 32x56x56xf32) <- (32x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_2, parameter_143, parameter_142, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_142, parameter_143

        # pd_op.matmul: (32x56x56x384xf32) <- (32x56x56x96xf32, 96x384xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_3, parameter_141, False, False)
        del parameter_141

        # pd_op.add: (32x56x56x384xf32) <- (32x56x56x384xf32, 384xf32)
        add_8 = paddle._C_ops.add(matmul_2, parameter_140)
        del parameter_140

        # pd_op.gelu: (32x56x56x384xf32) <- (32x56x56x384xf32)
        gelu_1 = paddle._C_ops.gelu(add_8, False)

        # pd_op.matmul: (32x56x56x96xf32) <- (32x56x56x384xf32, 384x96xf32)
        matmul_3 = paddle._C_ops.matmul(gelu_1, parameter_139, False, False)
        del parameter_139

        # pd_op.add: (32x56x56x96xf32) <- (32x56x56x96xf32, 96xf32)
        add_9 = paddle._C_ops.add(matmul_3, parameter_138)
        del parameter_138

        # pd_op.multiply: (32x56x56x96xf32) <- (96xf32, 32x56x56x96xf32)
        multiply_2 = paddle._C_ops.multiply(data_19, add_9)
        del data_19

        # pd_op.transpose: (32x96x56x56xf32) <- (32x56x56x96xf32)
        transpose_3 = paddle._C_ops.transpose(multiply_2, [0, 3, 1, 2])
        del multiply_2

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_1,
            [],
            paddle.float32,
            [float("0.994118")],
            paddle.framework._current_expected_place(),
        )
        del full_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [32, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_10 = paddle._C_ops.add(assign_value__0, uniform_0)
        del uniform_0

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (32x96x56x56xf32) <- (32x96x56x56xf32, xf32)
        divide_1 = paddle._C_ops.divide(transpose_3, assign_value__0)

        # pd_op.multiply: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_1, floor_0)

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x96x56x56xf32)
        add_11 = paddle._C_ops.add(add_6, multiply_3)

        # pd_op.depthwise_conv2d: (32x96x56x56xf32) <- (32x96x56x56xf32, 96x1x7x7xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            add_11, parameter_137, [1, 1], [3, 3], "EXPLICIT", 96, [1, 1], "NCHW"
        )
        del parameter_137

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_136, full_int_array_0)
        del parameter_136

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 1x96x1x1xf32)
        add_12 = paddle._C_ops.add(depthwise_conv2d_2, reshape_3)

        # pd_op.transpose: (32x56x56x96xf32) <- (32x96x56x56xf32)
        transpose_4 = paddle._C_ops.transpose(add_12, [0, 2, 3, 1])
        del add_12

        # pd_op.layer_norm: (32x56x56x96xf32, 32x56x56xf32, 32x56x56xf32) <- (32x56x56x96xf32, 96xf32, 96xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_4, parameter_135, parameter_134, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_134, parameter_135

        # pd_op.matmul: (32x56x56x384xf32) <- (32x56x56x96xf32, 96x384xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_6, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (32x56x56x384xf32) <- (32x56x56x384xf32, 384xf32)
        add_13 = paddle._C_ops.add(matmul_4, parameter_132)
        del parameter_132

        # pd_op.gelu: (32x56x56x384xf32) <- (32x56x56x384xf32)
        gelu_2 = paddle._C_ops.gelu(add_13, False)

        # pd_op.matmul: (32x56x56x96xf32) <- (32x56x56x384xf32, 384x96xf32)
        matmul_5 = paddle._C_ops.matmul(gelu_2, parameter_131, False, False)
        del parameter_131

        # pd_op.add: (32x56x56x96xf32) <- (32x56x56x96xf32, 96xf32)
        add_14 = paddle._C_ops.add(matmul_5, parameter_130)
        del parameter_130

        # pd_op.multiply: (32x56x56x96xf32) <- (96xf32, 32x56x56x96xf32)
        multiply_4 = paddle._C_ops.multiply(data_20, add_14)
        del data_20

        # pd_op.transpose: (32x96x56x56xf32) <- (32x56x56x96xf32)
        transpose_5 = paddle._C_ops.transpose(multiply_4, [0, 3, 1, 2])
        del multiply_4

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_3,
            [],
            paddle.float32,
            [float("0.988235")],
            paddle.framework._current_expected_place(),
        )
        del full_3

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_15 = paddle._C_ops.add(assign_value__1, uniform_1)
        del uniform_1

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_15)
        del add_15

        # pd_op.divide: (32x96x56x56xf32) <- (32x96x56x56xf32, xf32)
        divide_2 = paddle._C_ops.divide(transpose_5, assign_value__1)

        # pd_op.multiply: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_2, floor_1)

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x96x56x56xf32)
        add_16 = paddle._C_ops.add(add_11, multiply_5)

        # pd_op.mean: (32x1x56x56xf32) <- (32x96x56x56xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(add_16, full_int_array_1, True)

        # pd_op.subtract: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x56x56xf32)
        subtract_1 = paddle._C_ops.subtract(add_16, mean_2)

        # pd_op.pow: (32x96x56x56xf32) <- (32x96x56x56xf32)
        pow_1 = paddle._C_ops.pow(subtract_1, float("2"))

        # pd_op.mean: (32x1x56x56xf32) <- (32x96x56x56xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(pow_1, full_int_array_1, True)

        # pd_op.subtract: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x56x56xf32)
        subtract_2 = paddle._C_ops.subtract(add_16, mean_2)

        # pd_op.scale: (32x1x56x56xf32) <- (32x1x56x56xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(mean_3, full_0, float("1e-06"), True)
        del mean_3

        # pd_op.sqrt: (32x1x56x56xf32) <- (32x1x56x56xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_1)
        del scale_1

        # pd_op.divide: (32x96x56x56xf32) <- (32x96x56x56xf32, 32x1x56x56xf32)
        divide_3 = paddle._C_ops.divide(subtract_2, sqrt_1)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_21, full_int_array_2)
        del data_21

        # pd_op.multiply: (32x96x56x56xf32) <- (96x1x1xf32, 32x96x56x56xf32)
        multiply_6 = paddle._C_ops.multiply(unsqueeze_2, divide_3)

        # pd_op.unsqueeze: (96x1x1xf32) <- (96xf32, 2xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(data_22, full_int_array_2)
        del data_22

        # pd_op.add: (32x96x56x56xf32) <- (32x96x56x56xf32, 96x1x1xf32)
        add_17 = paddle._C_ops.add(multiply_6, unsqueeze_3)

        # pd_op.conv2d: (32x192x28x28xf32) <- (32x96x56x56xf32, 192x96x2x2xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            add_17, parameter_129, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_129

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_128, full_int_array_0)
        del parameter_128

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 1x192x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_1, reshape_4)

        # pd_op.depthwise_conv2d: (32x192x28x28xf32) <- (32x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            add_18, parameter_127, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_127

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(parameter_126, full_int_array_0)
        del parameter_126

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 1x192x1x1xf32)
        add_19 = paddle._C_ops.add(depthwise_conv2d_3, reshape_5)

        # pd_op.transpose: (32x28x28x192xf32) <- (32x192x28x28xf32)
        transpose_6 = paddle._C_ops.transpose(add_19, [0, 2, 3, 1])
        del add_19

        # pd_op.layer_norm: (32x28x28x192xf32, 32x28x28xf32, 32x28x28xf32) <- (32x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_6, parameter_125, parameter_124, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_124, parameter_125

        # pd_op.matmul: (32x28x28x768xf32) <- (32x28x28x192xf32, 192x768xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (32x28x28x768xf32) <- (32x28x28x768xf32, 768xf32)
        add_20 = paddle._C_ops.add(matmul_6, parameter_122)
        del parameter_122

        # pd_op.gelu: (32x28x28x768xf32) <- (32x28x28x768xf32)
        gelu_3 = paddle._C_ops.gelu(add_20, False)

        # pd_op.matmul: (32x28x28x192xf32) <- (32x28x28x768xf32, 768x192xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_3, parameter_121, False, False)
        del parameter_121

        # pd_op.add: (32x28x28x192xf32) <- (32x28x28x192xf32, 192xf32)
        add_21 = paddle._C_ops.add(matmul_7, parameter_120)
        del parameter_120

        # pd_op.multiply: (32x28x28x192xf32) <- (192xf32, 32x28x28x192xf32)
        multiply_7 = paddle._C_ops.multiply(data_23, add_21)
        del data_23

        # pd_op.transpose: (32x192x28x28xf32) <- (32x28x28x192xf32)
        transpose_7 = paddle._C_ops.transpose(multiply_7, [0, 3, 1, 2])
        del multiply_7

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_4,
            [],
            paddle.float32,
            [float("0.982353")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_22 = paddle._C_ops.add(assign_value__2, uniform_2)
        del uniform_2

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (32x192x28x28xf32) <- (32x192x28x28xf32, xf32)
        divide_4 = paddle._C_ops.divide(transpose_7, assign_value__2)

        # pd_op.multiply: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_4, floor_2)

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x192x28x28xf32)
        add_23 = paddle._C_ops.add(add_18, multiply_8)

        # pd_op.depthwise_conv2d: (32x192x28x28xf32) <- (32x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            add_23, parameter_119, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_119

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_118, full_int_array_0)
        del parameter_118

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 1x192x1x1xf32)
        add_24 = paddle._C_ops.add(depthwise_conv2d_4, reshape_6)

        # pd_op.transpose: (32x28x28x192xf32) <- (32x192x28x28xf32)
        transpose_8 = paddle._C_ops.transpose(add_24, [0, 2, 3, 1])
        del add_24

        # pd_op.layer_norm: (32x28x28x192xf32, 32x28x28xf32, 32x28x28xf32) <- (32x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_8, parameter_117, parameter_116, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_116, parameter_117

        # pd_op.matmul: (32x28x28x768xf32) <- (32x28x28x192xf32, 192x768xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_12, parameter_115, False, False)
        del parameter_115

        # pd_op.add: (32x28x28x768xf32) <- (32x28x28x768xf32, 768xf32)
        add_25 = paddle._C_ops.add(matmul_8, parameter_114)
        del parameter_114

        # pd_op.gelu: (32x28x28x768xf32) <- (32x28x28x768xf32)
        gelu_4 = paddle._C_ops.gelu(add_25, False)

        # pd_op.matmul: (32x28x28x192xf32) <- (32x28x28x768xf32, 768x192xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_4, parameter_113, False, False)
        del parameter_113

        # pd_op.add: (32x28x28x192xf32) <- (32x28x28x192xf32, 192xf32)
        add_26 = paddle._C_ops.add(matmul_9, parameter_112)
        del parameter_112

        # pd_op.multiply: (32x28x28x192xf32) <- (192xf32, 32x28x28x192xf32)
        multiply_9 = paddle._C_ops.multiply(data_24, add_26)
        del data_24

        # pd_op.transpose: (32x192x28x28xf32) <- (32x28x28x192xf32)
        transpose_9 = paddle._C_ops.transpose(multiply_9, [0, 3, 1, 2])
        del multiply_9

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_5,
            [],
            paddle.float32,
            [float("0.976471")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_27 = paddle._C_ops.add(assign_value__3, uniform_3)
        del uniform_3

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_27)
        del add_27

        # pd_op.divide: (32x192x28x28xf32) <- (32x192x28x28xf32, xf32)
        divide_5 = paddle._C_ops.divide(transpose_9, assign_value__3)

        # pd_op.multiply: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_5, floor_3)

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x192x28x28xf32)
        add_28 = paddle._C_ops.add(add_23, multiply_10)

        # pd_op.depthwise_conv2d: (32x192x28x28xf32) <- (32x192x28x28xf32, 192x1x7x7xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            add_28, parameter_111, [1, 1], [3, 3], "EXPLICIT", 192, [1, 1], "NCHW"
        )
        del parameter_111

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(parameter_110, full_int_array_0)
        del parameter_110

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 1x192x1x1xf32)
        add_29 = paddle._C_ops.add(depthwise_conv2d_5, reshape_7)

        # pd_op.transpose: (32x28x28x192xf32) <- (32x192x28x28xf32)
        transpose_10 = paddle._C_ops.transpose(add_29, [0, 2, 3, 1])
        del add_29

        # pd_op.layer_norm: (32x28x28x192xf32, 32x28x28xf32, 32x28x28xf32) <- (32x28x28x192xf32, 192xf32, 192xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_10, parameter_109, parameter_108, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_108, parameter_109

        # pd_op.matmul: (32x28x28x768xf32) <- (32x28x28x192xf32, 192x768xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_15, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (32x28x28x768xf32) <- (32x28x28x768xf32, 768xf32)
        add_30 = paddle._C_ops.add(matmul_10, parameter_106)
        del parameter_106

        # pd_op.gelu: (32x28x28x768xf32) <- (32x28x28x768xf32)
        gelu_5 = paddle._C_ops.gelu(add_30, False)

        # pd_op.matmul: (32x28x28x192xf32) <- (32x28x28x768xf32, 768x192xf32)
        matmul_11 = paddle._C_ops.matmul(gelu_5, parameter_105, False, False)
        del parameter_105

        # pd_op.add: (32x28x28x192xf32) <- (32x28x28x192xf32, 192xf32)
        add_31 = paddle._C_ops.add(matmul_11, parameter_104)
        del parameter_104

        # pd_op.multiply: (32x28x28x192xf32) <- (192xf32, 32x28x28x192xf32)
        multiply_11 = paddle._C_ops.multiply(data_25, add_31)
        del data_25

        # pd_op.transpose: (32x192x28x28xf32) <- (32x28x28x192xf32)
        transpose_11 = paddle._C_ops.transpose(multiply_11, [0, 3, 1, 2])
        del multiply_11

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.float32,
            [float("0.970588")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_32 = paddle._C_ops.add(assign_value__4, uniform_4)
        del uniform_4

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_32)
        del add_32

        # pd_op.divide: (32x192x28x28xf32) <- (32x192x28x28xf32, xf32)
        divide_6 = paddle._C_ops.divide(transpose_11, assign_value__4)

        # pd_op.multiply: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_6, floor_4)

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x192x28x28xf32)
        add_33 = paddle._C_ops.add(add_28, multiply_12)

        # pd_op.mean: (32x1x28x28xf32) <- (32x192x28x28xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(add_33, full_int_array_1, True)

        # pd_op.subtract: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x28x28xf32)
        subtract_3 = paddle._C_ops.subtract(add_33, mean_4)

        # pd_op.pow: (32x192x28x28xf32) <- (32x192x28x28xf32)
        pow_2 = paddle._C_ops.pow(subtract_3, float("2"))

        # pd_op.mean: (32x1x28x28xf32) <- (32x192x28x28xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(pow_2, full_int_array_1, True)

        # pd_op.subtract: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x28x28xf32)
        subtract_4 = paddle._C_ops.subtract(add_33, mean_4)

        # pd_op.scale: (32x1x28x28xf32) <- (32x1x28x28xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(mean_5, full_0, float("1e-06"), True)
        del mean_5

        # pd_op.sqrt: (32x1x28x28xf32) <- (32x1x28x28xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_2)
        del scale_2

        # pd_op.divide: (32x192x28x28xf32) <- (32x192x28x28xf32, 32x1x28x28xf32)
        divide_7 = paddle._C_ops.divide(subtract_4, sqrt_2)

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(data_2, full_int_array_2)
        del data_2

        # pd_op.multiply: (32x192x28x28xf32) <- (192x1x1xf32, 32x192x28x28xf32)
        multiply_13 = paddle._C_ops.multiply(unsqueeze_4, divide_7)

        # pd_op.unsqueeze: (192x1x1xf32) <- (192xf32, 2xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(data_3, full_int_array_2)
        del data_3

        # pd_op.add: (32x192x28x28xf32) <- (32x192x28x28xf32, 192x1x1xf32)
        add_34 = paddle._C_ops.add(multiply_13, unsqueeze_5)

        # pd_op.conv2d: (32x384x14x14xf32) <- (32x192x28x28xf32, 384x192x2x2xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_34, parameter_103, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_103

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_102, full_int_array_0)
        del parameter_102

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_2, reshape_8)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            add_35, parameter_101, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_101

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(parameter_100, full_int_array_0)
        del parameter_100

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_36 = paddle._C_ops.add(depthwise_conv2d_6, reshape_9)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_12 = paddle._C_ops.transpose(add_36, [0, 2, 3, 1])
        del add_36

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_12, parameter_99, parameter_98, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_98, parameter_99

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_18, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_37 = paddle._C_ops.add(matmul_12, parameter_96)
        del parameter_96

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_6 = paddle._C_ops.gelu(add_37, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_13 = paddle._C_ops.matmul(gelu_6, parameter_95, False, False)
        del parameter_95

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_38 = paddle._C_ops.add(matmul_13, parameter_94)
        del parameter_94

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_14 = paddle._C_ops.multiply(data_4, add_38)
        del data_4

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_13 = paddle._C_ops.transpose(multiply_14, [0, 3, 1, 2])
        del multiply_14

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_7,
            [],
            paddle.float32,
            [float("0.964706")],
            paddle.framework._current_expected_place(),
        )
        del full_7

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_39 = paddle._C_ops.add(assign_value__5, uniform_5)
        del uniform_5

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_39)
        del add_39

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(transpose_13, assign_value__5)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_8, floor_5)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_40 = paddle._C_ops.add(add_35, multiply_15)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            add_40, parameter_93, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_93

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_92, full_int_array_0)
        del parameter_92

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_41 = paddle._C_ops.add(depthwise_conv2d_7, reshape_10)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_14 = paddle._C_ops.transpose(add_41, [0, 2, 3, 1])
        del add_41

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_14, parameter_91, parameter_90, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_90, parameter_91

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_21, parameter_89, False, False)
        del parameter_89

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_42 = paddle._C_ops.add(matmul_14, parameter_88)
        del parameter_88

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add_42, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_7, parameter_87, False, False)
        del parameter_87

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_43 = paddle._C_ops.add(matmul_15, parameter_86)
        del parameter_86

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_16 = paddle._C_ops.multiply(data_5, add_43)
        del data_5

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_15 = paddle._C_ops.transpose(multiply_16, [0, 3, 1, 2])
        del multiply_16

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_8,
            [],
            paddle.float32,
            [float("0.958824")],
            paddle.framework._current_expected_place(),
        )
        del full_8

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_44 = paddle._C_ops.add(assign_value__6, uniform_6)
        del uniform_6

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_44)
        del add_44

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(transpose_15, assign_value__6)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_9, floor_6)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_45 = paddle._C_ops.add(add_40, multiply_17)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            add_45, parameter_85, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_85

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(parameter_84, full_int_array_0)
        del parameter_84

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_46 = paddle._C_ops.add(depthwise_conv2d_8, reshape_11)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_16 = paddle._C_ops.transpose(add_46, [0, 2, 3, 1])
        del add_46

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_16, parameter_83, parameter_82, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_82, parameter_83

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_24, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_47 = paddle._C_ops.add(matmul_16, parameter_80)
        del parameter_80

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_8 = paddle._C_ops.gelu(add_47, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_17 = paddle._C_ops.matmul(gelu_8, parameter_79, False, False)
        del parameter_79

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_48 = paddle._C_ops.add(matmul_17, parameter_78)
        del parameter_78

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_18 = paddle._C_ops.multiply(data_6, add_48)
        del data_6

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_17 = paddle._C_ops.transpose(multiply_18, [0, 3, 1, 2])
        del multiply_18

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_9,
            [],
            paddle.float32,
            [float("0.952941")],
            paddle.framework._current_expected_place(),
        )
        del full_9

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_49 = paddle._C_ops.add(assign_value__7, uniform_7)
        del uniform_7

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_49)
        del add_49

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_10 = paddle._C_ops.divide(transpose_17, assign_value__7)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_10, floor_7)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_50 = paddle._C_ops.add(add_45, multiply_19)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            add_50, parameter_77, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_77

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_76, full_int_array_0)
        del parameter_76

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_51 = paddle._C_ops.add(depthwise_conv2d_9, reshape_12)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_18 = paddle._C_ops.transpose(add_51, [0, 2, 3, 1])
        del add_51

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_18, parameter_75, parameter_74, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_74, parameter_75

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_27, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_52 = paddle._C_ops.add(matmul_18, parameter_72)
        del parameter_72

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add_52, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_19 = paddle._C_ops.matmul(gelu_9, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_53 = paddle._C_ops.add(matmul_19, parameter_70)
        del parameter_70

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_20 = paddle._C_ops.multiply(data_7, add_53)
        del data_7

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_19 = paddle._C_ops.transpose(multiply_20, [0, 3, 1, 2])
        del multiply_20

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.float32,
            [float("0.947059")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_54 = paddle._C_ops.add(assign_value__8, uniform_8)
        del uniform_8

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_54)
        del add_54

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_11 = paddle._C_ops.divide(transpose_19, assign_value__8)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_21 = paddle._C_ops.multiply(divide_11, floor_8)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_55 = paddle._C_ops.add(add_50, multiply_21)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            add_55, parameter_69, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_69

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(parameter_68, full_int_array_0)
        del parameter_68

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_56 = paddle._C_ops.add(depthwise_conv2d_10, reshape_13)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_20 = paddle._C_ops.transpose(add_56, [0, 2, 3, 1])
        del add_56

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_20, parameter_67, parameter_66, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_66, parameter_67

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_30, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_57 = paddle._C_ops.add(matmul_20, parameter_64)
        del parameter_64

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_10 = paddle._C_ops.gelu(add_57, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_21 = paddle._C_ops.matmul(gelu_10, parameter_63, False, False)
        del parameter_63

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_58 = paddle._C_ops.add(matmul_21, parameter_62)
        del parameter_62

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_22 = paddle._C_ops.multiply(data_8, add_58)
        del data_8

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_21 = paddle._C_ops.transpose(multiply_22, [0, 3, 1, 2])
        del multiply_22

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_11,
            [],
            paddle.float32,
            [float("0.941176")],
            paddle.framework._current_expected_place(),
        )
        del full_11

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_59 = paddle._C_ops.add(assign_value__9, uniform_9)
        del uniform_9

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_59)
        del add_59

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_12 = paddle._C_ops.divide(transpose_21, assign_value__9)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_23 = paddle._C_ops.multiply(divide_12, floor_9)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_60 = paddle._C_ops.add(add_55, multiply_23)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            add_60, parameter_61, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_61

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_60, full_int_array_0)
        del parameter_60

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_61 = paddle._C_ops.add(depthwise_conv2d_11, reshape_14)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_22 = paddle._C_ops.transpose(add_61, [0, 2, 3, 1])
        del add_61

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_22, parameter_59, parameter_58, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_58, parameter_59

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_33, parameter_57, False, False)
        del parameter_57

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_62 = paddle._C_ops.add(matmul_22, parameter_56)
        del parameter_56

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_11 = paddle._C_ops.gelu(add_62, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_23 = paddle._C_ops.matmul(gelu_11, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_63 = paddle._C_ops.add(matmul_23, parameter_54)
        del parameter_54

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_24 = paddle._C_ops.multiply(data_9, add_63)
        del data_9

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_23 = paddle._C_ops.transpose(multiply_24, [0, 3, 1, 2])
        del multiply_24

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_12,
            [],
            paddle.float32,
            [float("0.935294")],
            paddle.framework._current_expected_place(),
        )
        del full_12

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_64 = paddle._C_ops.add(assign_value__10, uniform_10)
        del uniform_10

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_64)
        del add_64

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_13 = paddle._C_ops.divide(transpose_23, assign_value__10)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_25 = paddle._C_ops.multiply(divide_13, floor_10)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_65 = paddle._C_ops.add(add_60, multiply_25)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(
            add_65, parameter_53, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_53

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(parameter_52, full_int_array_0)
        del parameter_52

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_66 = paddle._C_ops.add(depthwise_conv2d_12, reshape_15)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_24 = paddle._C_ops.transpose(add_66, [0, 2, 3, 1])
        del add_66

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_24, parameter_51, parameter_50, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_50, parameter_51

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_36, parameter_49, False, False)
        del parameter_49

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_67 = paddle._C_ops.add(matmul_24, parameter_48)
        del parameter_48

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_12 = paddle._C_ops.gelu(add_67, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_25 = paddle._C_ops.matmul(gelu_12, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_68 = paddle._C_ops.add(matmul_25, parameter_46)
        del parameter_46

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_26 = paddle._C_ops.multiply(data_10, add_68)
        del data_10

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_25 = paddle._C_ops.transpose(multiply_26, [0, 3, 1, 2])
        del multiply_26

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.float32,
            [float("0.929412")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_69 = paddle._C_ops.add(assign_value__11, uniform_11)
        del uniform_11

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_69)
        del add_69

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_14 = paddle._C_ops.divide(transpose_25, assign_value__11)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_27 = paddle._C_ops.multiply(divide_14, floor_11)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_70 = paddle._C_ops.add(add_65, multiply_27)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(
            add_70, parameter_45, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_45

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_44, full_int_array_0)
        del parameter_44

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_71 = paddle._C_ops.add(depthwise_conv2d_13, reshape_16)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_26 = paddle._C_ops.transpose(add_71, [0, 2, 3, 1])
        del add_71

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_26, parameter_43, parameter_42, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_42, parameter_43

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_39, parameter_41, False, False)
        del parameter_41

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_72 = paddle._C_ops.add(matmul_26, parameter_40)
        del parameter_40

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_13 = paddle._C_ops.gelu(add_72, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_27 = paddle._C_ops.matmul(gelu_13, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_73 = paddle._C_ops.add(matmul_27, parameter_38)
        del parameter_38

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_28 = paddle._C_ops.multiply(data_11, add_73)
        del data_11

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_27 = paddle._C_ops.transpose(multiply_28, [0, 3, 1, 2])
        del multiply_28

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.float32,
            [float("0.923529")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_74 = paddle._C_ops.add(assign_value__12, uniform_12)
        del uniform_12

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_74)
        del add_74

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_15 = paddle._C_ops.divide(transpose_27, assign_value__12)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_29 = paddle._C_ops.multiply(divide_15, floor_12)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_75 = paddle._C_ops.add(add_70, multiply_29)

        # pd_op.depthwise_conv2d: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x7x7xf32)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(
            add_75, parameter_37, [1, 1], [3, 3], "EXPLICIT", 384, [1, 1], "NCHW"
        )
        del parameter_37

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(parameter_36, full_int_array_0)
        del parameter_36

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 1x384x1x1xf32)
        add_76 = paddle._C_ops.add(depthwise_conv2d_14, reshape_17)

        # pd_op.transpose: (32x14x14x384xf32) <- (32x384x14x14xf32)
        transpose_28 = paddle._C_ops.transpose(add_76, [0, 2, 3, 1])
        del add_76

        # pd_op.layer_norm: (32x14x14x384xf32, 32x14x14xf32, 32x14x14xf32) <- (32x14x14x384xf32, 384xf32, 384xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_28, parameter_35, parameter_34, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_34, parameter_35

        # pd_op.matmul: (32x14x14x1536xf32) <- (32x14x14x384xf32, 384x1536xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_42, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (32x14x14x1536xf32) <- (32x14x14x1536xf32, 1536xf32)
        add_77 = paddle._C_ops.add(matmul_28, parameter_32)
        del parameter_32

        # pd_op.gelu: (32x14x14x1536xf32) <- (32x14x14x1536xf32)
        gelu_14 = paddle._C_ops.gelu(add_77, False)

        # pd_op.matmul: (32x14x14x384xf32) <- (32x14x14x1536xf32, 1536x384xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_14, parameter_31, False, False)
        del parameter_31

        # pd_op.add: (32x14x14x384xf32) <- (32x14x14x384xf32, 384xf32)
        add_78 = paddle._C_ops.add(matmul_29, parameter_30)
        del parameter_30

        # pd_op.multiply: (32x14x14x384xf32) <- (384xf32, 32x14x14x384xf32)
        multiply_30 = paddle._C_ops.multiply(data_13, add_78)
        del data_13

        # pd_op.transpose: (32x384x14x14xf32) <- (32x14x14x384xf32)
        transpose_29 = paddle._C_ops.transpose(multiply_30, [0, 3, 1, 2])
        del multiply_30

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.float32,
            [float("0.917647")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_79 = paddle._C_ops.add(assign_value__13, uniform_13)
        del uniform_13

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_79)
        del add_79

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, xf32)
        divide_16 = paddle._C_ops.divide(transpose_29, assign_value__13)

        # pd_op.multiply: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x1x1xf32)
        multiply_31 = paddle._C_ops.multiply(divide_16, floor_13)

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x384x14x14xf32)
        add_80 = paddle._C_ops.add(add_75, multiply_31)

        # pd_op.mean: (32x1x14x14xf32) <- (32x384x14x14xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(add_80, full_int_array_1, True)

        # pd_op.subtract: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x14x14xf32)
        subtract_5 = paddle._C_ops.subtract(add_80, mean_6)

        # pd_op.pow: (32x384x14x14xf32) <- (32x384x14x14xf32)
        pow_3 = paddle._C_ops.pow(subtract_5, float("2"))

        # pd_op.mean: (32x1x14x14xf32) <- (32x384x14x14xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(pow_3, full_int_array_1, True)

        # pd_op.subtract: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x14x14xf32)
        subtract_6 = paddle._C_ops.subtract(add_80, mean_6)

        # pd_op.scale: (32x1x14x14xf32) <- (32x1x14x14xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(mean_7, full_0, float("1e-06"), True)
        del mean_7

        # pd_op.sqrt: (32x1x14x14xf32) <- (32x1x14x14xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_3)
        del scale_3

        # pd_op.divide: (32x384x14x14xf32) <- (32x384x14x14xf32, 32x1x14x14xf32)
        divide_17 = paddle._C_ops.divide(subtract_6, sqrt_3)

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(data_14, full_int_array_2)
        del data_14

        # pd_op.multiply: (32x384x14x14xf32) <- (384x1x1xf32, 32x384x14x14xf32)
        multiply_32 = paddle._C_ops.multiply(unsqueeze_6, divide_17)

        # pd_op.unsqueeze: (384x1x1xf32) <- (384xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(data_15, full_int_array_2)
        del data_15

        # pd_op.add: (32x384x14x14xf32) <- (32x384x14x14xf32, 384x1x1xf32)
        add_81 = paddle._C_ops.add(multiply_32, unsqueeze_7)

        # pd_op.conv2d: (32x768x7x7xf32) <- (32x384x14x14xf32, 768x384x2x2xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_81, parameter_29, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_28, full_int_array_0)
        del parameter_28

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 1x768x1x1xf32)
        add_82 = paddle._C_ops.add(conv2d_3, reshape_18)

        # pd_op.depthwise_conv2d: (32x768x7x7xf32) <- (32x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(
            add_82, parameter_27, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_27

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(parameter_26, full_int_array_0)
        del parameter_26

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 1x768x1x1xf32)
        add_83 = paddle._C_ops.add(depthwise_conv2d_15, reshape_19)

        # pd_op.transpose: (32x7x7x768xf32) <- (32x768x7x7xf32)
        transpose_30 = paddle._C_ops.transpose(add_83, [0, 2, 3, 1])
        del add_83

        # pd_op.layer_norm: (32x7x7x768xf32, 32x7x7xf32, 32x7x7xf32) <- (32x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_30, parameter_25, parameter_24, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_24, parameter_25

        # pd_op.matmul: (32x7x7x3072xf32) <- (32x7x7x768xf32, 768x3072xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_45, parameter_23, False, False)
        del parameter_23

        # pd_op.add: (32x7x7x3072xf32) <- (32x7x7x3072xf32, 3072xf32)
        add_84 = paddle._C_ops.add(matmul_30, parameter_22)
        del parameter_22

        # pd_op.gelu: (32x7x7x3072xf32) <- (32x7x7x3072xf32)
        gelu_15 = paddle._C_ops.gelu(add_84, False)

        # pd_op.matmul: (32x7x7x768xf32) <- (32x7x7x3072xf32, 3072x768xf32)
        matmul_31 = paddle._C_ops.matmul(gelu_15, parameter_21, False, False)
        del parameter_21

        # pd_op.add: (32x7x7x768xf32) <- (32x7x7x768xf32, 768xf32)
        add_85 = paddle._C_ops.add(matmul_31, parameter_20)
        del parameter_20

        # pd_op.multiply: (32x7x7x768xf32) <- (768xf32, 32x7x7x768xf32)
        multiply_33 = paddle._C_ops.multiply(data_16, add_85)
        del data_16

        # pd_op.transpose: (32x768x7x7xf32) <- (32x7x7x768xf32)
        transpose_31 = paddle._C_ops.transpose(multiply_33, [0, 3, 1, 2])
        del multiply_33

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__14 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.float32,
            [float("0.911765")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_86 = paddle._C_ops.add(assign_value__14, uniform_14)
        del uniform_14

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_86)
        del add_86

        # pd_op.divide: (32x768x7x7xf32) <- (32x768x7x7xf32, xf32)
        divide_18 = paddle._C_ops.divide(transpose_31, assign_value__14)

        # pd_op.multiply: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x1x1x1xf32)
        multiply_34 = paddle._C_ops.multiply(divide_18, floor_14)

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x768x7x7xf32)
        add_87 = paddle._C_ops.add(add_82, multiply_34)

        # pd_op.depthwise_conv2d: (32x768x7x7xf32) <- (32x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(
            add_87, parameter_19, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_19

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_18, full_int_array_0)
        del parameter_18

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 1x768x1x1xf32)
        add_88 = paddle._C_ops.add(depthwise_conv2d_16, reshape_20)

        # pd_op.transpose: (32x7x7x768xf32) <- (32x768x7x7xf32)
        transpose_32 = paddle._C_ops.transpose(add_88, [0, 2, 3, 1])
        del add_88

        # pd_op.layer_norm: (32x7x7x768xf32, 32x7x7xf32, 32x7x7xf32) <- (32x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_32, parameter_17, parameter_16, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_16, parameter_17

        # pd_op.matmul: (32x7x7x3072xf32) <- (32x7x7x768xf32, 768x3072xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_48, parameter_15, False, False)
        del parameter_15

        # pd_op.add: (32x7x7x3072xf32) <- (32x7x7x3072xf32, 3072xf32)
        add_89 = paddle._C_ops.add(matmul_32, parameter_14)
        del parameter_14

        # pd_op.gelu: (32x7x7x3072xf32) <- (32x7x7x3072xf32)
        gelu_16 = paddle._C_ops.gelu(add_89, False)

        # pd_op.matmul: (32x7x7x768xf32) <- (32x7x7x3072xf32, 3072x768xf32)
        matmul_33 = paddle._C_ops.matmul(gelu_16, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (32x7x7x768xf32) <- (32x7x7x768xf32, 768xf32)
        add_90 = paddle._C_ops.add(matmul_33, parameter_12)
        del parameter_12

        # pd_op.multiply: (32x7x7x768xf32) <- (768xf32, 32x7x7x768xf32)
        multiply_35 = paddle._C_ops.multiply(data_17, add_90)
        del data_17

        # pd_op.transpose: (32x768x7x7xf32) <- (32x7x7x768xf32)
        transpose_33 = paddle._C_ops.transpose(multiply_35, [0, 3, 1, 2])
        del multiply_35

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__15 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.float32,
            [float("0.905882")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_91 = paddle._C_ops.add(assign_value__15, uniform_15)
        del uniform_15

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_91)
        del add_91

        # pd_op.divide: (32x768x7x7xf32) <- (32x768x7x7xf32, xf32)
        divide_19 = paddle._C_ops.divide(transpose_33, assign_value__15)

        # pd_op.multiply: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x1x1x1xf32)
        multiply_36 = paddle._C_ops.multiply(divide_19, floor_15)

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x768x7x7xf32)
        add_92 = paddle._C_ops.add(add_87, multiply_36)

        # pd_op.depthwise_conv2d: (32x768x7x7xf32) <- (32x768x7x7xf32, 768x1x7x7xf32)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(
            add_92, parameter_11, [1, 1], [3, 3], "EXPLICIT", 768, [1, 1], "NCHW"
        )
        del parameter_11

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(parameter_10, full_int_array_0)
        del full_int_array_0, parameter_10

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 1x768x1x1xf32)
        add_93 = paddle._C_ops.add(depthwise_conv2d_17, reshape_21)

        # pd_op.transpose: (32x7x7x768xf32) <- (32x768x7x7xf32)
        transpose_34 = paddle._C_ops.transpose(add_93, [0, 2, 3, 1])
        del add_93

        # pd_op.layer_norm: (32x7x7x768xf32, 32x7x7xf32, 32x7x7xf32) <- (32x7x7x768xf32, 768xf32, 768xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                transpose_34, parameter_9, parameter_8, float("1e-06"), 3
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_8, parameter_9

        # pd_op.matmul: (32x7x7x3072xf32) <- (32x7x7x768xf32, 768x3072xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_51, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (32x7x7x3072xf32) <- (32x7x7x3072xf32, 3072xf32)
        add_94 = paddle._C_ops.add(matmul_34, parameter_6)
        del parameter_6

        # pd_op.gelu: (32x7x7x3072xf32) <- (32x7x7x3072xf32)
        gelu_17 = paddle._C_ops.gelu(add_94, False)

        # pd_op.matmul: (32x7x7x768xf32) <- (32x7x7x3072xf32, 3072x768xf32)
        matmul_35 = paddle._C_ops.matmul(gelu_17, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (32x7x7x768xf32) <- (32x7x7x768xf32, 768xf32)
        add_95 = paddle._C_ops.add(matmul_35, parameter_4)
        del parameter_4

        # pd_op.multiply: (32x7x7x768xf32) <- (768xf32, 32x7x7x768xf32)
        multiply_37 = paddle._C_ops.multiply(data_18, add_95)
        del data_18

        # pd_op.transpose: (32x768x7x7xf32) <- (32x7x7x768xf32)
        transpose_35 = paddle._C_ops.transpose(multiply_37, [0, 3, 1, 2])
        del multiply_37

        # pd_op.full: (xf32) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xf32) <- (xf32)
        assign_value__16 = paddle._C_ops.assign_value_(
            full_18,
            [],
            paddle.float32,
            [float("0.9")],
            paddle.framework._current_expected_place(),
        )
        del full_18

        # pd_op.uniform: (32x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_2,
            full_0,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_2, full_int_array_3

        # pd_op.add: (32x1x1x1xf32) <- (xf32, 32x1x1x1xf32)
        add_96 = paddle._C_ops.add(assign_value__16, uniform_16)
        del uniform_16

        # pd_op.floor: (32x1x1x1xf32) <- (32x1x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_96)
        del add_96

        # pd_op.divide: (32x768x7x7xf32) <- (32x768x7x7xf32, xf32)
        divide_20 = paddle._C_ops.divide(transpose_35, assign_value__16)

        # pd_op.multiply: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x1x1x1xf32)
        multiply_38 = paddle._C_ops.multiply(divide_20, floor_16)

        # pd_op.add: (32x768x7x7xf32) <- (32x768x7x7xf32, 32x768x7x7xf32)
        add_97 = paddle._C_ops.add(add_92, multiply_38)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-2, -1]

        # pd_op.mean: (32x768xf32) <- (32x768x7x7xf32, 2xi64)
        mean_8 = paddle._C_ops.mean(add_97, full_int_array_4, False)

        # pd_op.layer_norm: (32x768xf32, 32xf32, 32xf32) <- (32x768xf32, 768xf32, 768xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                mean_8, parameter_3, parameter_2, float("1e-06"), 1
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_2, parameter_3

        # pd_op.matmul: (32x102xf32) <- (32x768xf32, 768x102xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_54, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (32x102xf32) <- (32x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_36, parameter_0)
        del (
            add_1,
            add_11,
            add_13,
            add_14,
            add_16,
            add_17,
            add_18,
            add_2,
            add_20,
            add_21,
            add_23,
            add_25,
            add_26,
            add_28,
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
            add_45,
            add_47,
            add_48,
            add_5,
            add_50,
            add_52,
            add_53,
            add_55,
            add_57,
            add_58,
            add_6,
            add_60,
            add_62,
            add_63,
            add_65,
            add_67,
            add_68,
            add_70,
            add_72,
            add_73,
            add_75,
            add_77,
            add_78,
            add_8,
            add_80,
            add_81,
            add_82,
            add_84,
            add_85,
            add_87,
            add_89,
            add_9,
            add_90,
            add_92,
            add_94,
            add_95,
            add_97,
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
            assign_value__0,
            assign_value__1,
            assign_value__10,
            assign_value__11,
            assign_value__12,
            assign_value__13,
            assign_value__14,
            assign_value__15,
            assign_value__16,
            assign_value__2,
            assign_value__3,
            assign_value__4,
            assign_value__5,
            assign_value__6,
            assign_value__7,
            assign_value__8,
            assign_value__9,
            conv2d_0,
            conv2d_1,
            conv2d_2,
            conv2d_3,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_10,
            depthwise_conv2d_11,
            depthwise_conv2d_12,
            depthwise_conv2d_13,
            depthwise_conv2d_14,
            depthwise_conv2d_15,
            depthwise_conv2d_16,
            depthwise_conv2d_17,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
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
            floor_12,
            floor_13,
            floor_14,
            floor_15,
            floor_16,
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
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_12,
            gelu_13,
            gelu_14,
            gelu_15,
            gelu_16,
            gelu_17,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
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
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_0,
            matmul_1,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_13,
            matmul_14,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
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
            matmul_4,
            matmul_5,
            matmul_6,
            matmul_7,
            matmul_8,
            matmul_9,
            mean_0,
            mean_2,
            mean_4,
            mean_6,
            mean_8,
            multiply_0,
            multiply_10,
            multiply_12,
            multiply_13,
            multiply_15,
            multiply_17,
            multiply_19,
            multiply_21,
            multiply_23,
            multiply_25,
            multiply_27,
            multiply_29,
            multiply_3,
            multiply_31,
            multiply_32,
            multiply_34,
            multiply_36,
            multiply_38,
            multiply_5,
            multiply_6,
            multiply_8,
            parameter_0,
            pow_0,
            pow_1,
            pow_2,
            pow_3,
            reshape_0,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_12,
            reshape_13,
            reshape_14,
            reshape_15,
            reshape_16,
            reshape_17,
            reshape_18,
            reshape_19,
            reshape_2,
            reshape_20,
            reshape_21,
            reshape_3,
            reshape_4,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            sqrt_0,
            sqrt_1,
            sqrt_2,
            sqrt_3,
            subtract_0,
            subtract_1,
            subtract_2,
            subtract_3,
            subtract_4,
            subtract_5,
            subtract_6,
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
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
            unsqueeze_1,
            unsqueeze_2,
            unsqueeze_3,
            unsqueeze_4,
            unsqueeze_5,
            unsqueeze_6,
            unsqueeze_7,
        )

        return add_0
