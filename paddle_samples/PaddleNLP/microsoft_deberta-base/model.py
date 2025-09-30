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
        parameter_181,
        parameter_182,
        parameter_183,
        parameter_184,
        parameter_185,
        parameter_186,
        parameter_187,
        parameter_188,
        parameter_189,
        parameter_190,
        parameter_191,
        parameter_192,
        parameter_193,
        parameter_194,
        parameter_195,
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("21"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (21xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="int64")
        del full_0, full_1, full_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x21xi64) <- (21xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(arange_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 21]

        # pd_op.expand: (1x21xi64) <- (1x21xi64, 2xi64)
        expand_0 = paddle._C_ops.expand(unsqueeze_0, full_int_array_1)
        del full_int_array_1

        # pd_op.embedding: (1x21x768xf32) <- (1x21xi64, 50265x768xf32)
        embedding_0 = paddle._C_ops.embedding(data_0, parameter_195, 0, False)
        del data_0, parameter_195

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1x21x768xf32) <- (1x21x768xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            embedding_0,
            full_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-1]

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_0 = paddle._C_ops.mean(embedding_0, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_0 = paddle._C_ops.subtract(embedding_0, mean_0)
        del embedding_0, mean_0

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_0 = paddle._C_ops.pow(subtract_0, float("2"))

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_1 = paddle._C_ops.mean(pow_0, full_int_array_2, True)
        del pow_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(mean_1, full_4, float("1e-07"), True)
        del mean_1

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_0 = paddle._C_ops.sqrt(scale_0)
        del scale_0

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_0 = paddle._C_ops.divide(subtract_0, sqrt_0)
        del sqrt_0, subtract_0

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_0 = paddle._C_ops.multiply(parameter_194, divide_0)
        del divide_0, parameter_194

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_1 = paddle._C_ops.add(multiply_0, parameter_193)
        del multiply_0, parameter_193

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.unsqueeze: (1x21x1xi64) <- (1x21xi64, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_1, full_int_array_3)

        # pd_op.cast: (1x21x1xf32) <- (1x21x1xi64)
        cast_0 = paddle._C_ops.cast(unsqueeze_1, paddle.float32)
        del unsqueeze_1

        # pd_op.multiply: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        multiply_1 = paddle._C_ops.multiply(add_1, cast_0)
        del add_1, cast_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.unsqueeze: (1x1x21xi64) <- (1x21xi64, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(data_1, full_int_array_4)
        del data_1

        # pd_op.unsqueeze: (1x1x1x21xi64) <- (1x1x21xi64, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(unsqueeze_2, full_int_array_3)
        del full_int_array_3, unsqueeze_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-2]

        # pd_op.squeeze: (1x1x21xi64) <- (1x1x1x21xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(unsqueeze_3, full_int_array_5)
        del full_int_array_5

        # pd_op.unsqueeze: (1x1x21x1xi64) <- (1x1x21xi64, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(squeeze_0, full_int_array_2)
        del squeeze_0

        # pd_op.multiply: (1x1x21x21xi64) <- (1x1x1x21xi64, 1x1x21x1xi64)
        multiply_2 = paddle._C_ops.multiply(unsqueeze_3, unsqueeze_4)
        del unsqueeze_3, unsqueeze_4

        # pd_op.cast: (1x1x21x21xf32) <- (1x1x21x21xi64)
        cast_1 = paddle._C_ops.cast(multiply_2, paddle.float32)
        del multiply_2

        # pd_op.unsqueeze: (21x1xi64) <- (21xi64, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(arange_0, full_int_array_4)
        del arange_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [21, 1]

        # pd_op.tile: (21x21xi64) <- (1x21xi64, 2xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_6)
        del full_int_array_6, unsqueeze_0

        # pd_op.subtract: (21x21xi64) <- (21x1xi64, 21x21xi64)
        subtract_1 = paddle._C_ops.subtract(unsqueeze_5, tile_0)
        del tile_0, unsqueeze_5

        # pd_op.unsqueeze: (1x21x21xi64) <- (21x21xi64, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(subtract_1, full_int_array_0)
        del subtract_1

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_0 = paddle._C_ops.matmul(multiply_1, parameter_190, False, False)
        del parameter_190

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 21, 12, -1]

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(matmul_0, full_int_array_7)
        del matmul_0

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(transpose_0, 3, full_5)
        del transpose_0

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_0,
            split_1,
            split_2,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [0, 1]

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(parameter_192, full_int_array_8)
        del parameter_192

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 1, 12, -1]

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(unsqueeze_7, full_int_array_9)
        del unsqueeze_7

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_2 = paddle._C_ops.add(split_0, transpose_1)
        del split_0, transpose_1

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(parameter_191, full_int_array_8)
        del parameter_191

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(unsqueeze_8, full_int_array_9)
        del unsqueeze_8

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_3 = paddle._C_ops.add(split_2, transpose_2)
        del split_2, transpose_2

        # pd_op.full: (xi64) <- ()
        full_6 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_6,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(assign_value__0, paddle.float32)
        del assign_value__0

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_2, full_7, float("0"), True)
        del cast_2

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_1 = paddle._C_ops.sqrt(scale_1)
        del scale_1

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_1 = paddle._C_ops.divide(add_2, sqrt_1)
        del add_2, sqrt_1

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_3 = paddle._C_ops.transpose(split_1, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_1 = paddle._C_ops.matmul(divide_1, transpose_3, False, False)
        del transpose_3

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.unsqueeze: (1x1x21x21xi64) <- (1x21x21xi64, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(unsqueeze_6, full_int_array_4)
        del full_int_array_4, unsqueeze_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [491]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [533]

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            dropout_0, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_0

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(slice_0, full_int_array_0)
        del slice_0

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_2 = paddle._C_ops.matmul(unsqueeze_10, parameter_189, False, False)
        del parameter_189

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 42, 12, -1]

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(matmul_2, full_int_array_12)
        del matmul_2

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_4 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_5 = paddle._C_ops.transpose(transpose_4, [0, 1, 3, 2])
        del transpose_4

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_3 = paddle._C_ops.matmul(divide_1, transpose_5, False, False)
        del divide_1, transpose_5

        # pd_op.scale: (1x1x21x21xi64) <- (1x1x21x21xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(unsqueeze_9, full_4, float("21"), True)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("41"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (1x1x21x21xi64) <- (1x1x21x21xi64, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(scale_2, full_3, full_9)
        del scale_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 12, 21, 21]

        # pd_op.expand: (1x12x21x21xi64) <- (1x1x21x21xi64, 4xi64)
        expand_1 = paddle._C_ops.expand(clip_0, full_int_array_13)
        del clip_0

        # pd_op.expand: (1x12x21x21xi64) <- (1x12x21x21xi64, 4xi64)
        expand_2 = paddle._C_ops.expand(expand_1, full_int_array_13)
        del expand_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 12, 21, 42]

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_3 = paddle._C_ops.expand(matmul_3, full_int_array_14)
        del matmul_3

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_0 = paddle._C_ops.take_along_axis(expand_3, expand_2, 3)
        del expand_3

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(take_along_axis_0, full_4, float("0"), True)
        del take_along_axis_0

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_4 = paddle._C_ops.matmul(unsqueeze_10, parameter_188, False, False)
        del parameter_188, unsqueeze_10

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_4 = paddle._C_ops.add(matmul_4, parameter_187)
        del matmul_4, parameter_187

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_4, full_int_array_12)
        del add_4

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.full: (xi64) <- ()
        full_10 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__1 = paddle._C_ops.assign_value_(
            full_10,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_10

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(assign_value__1, paddle.float32)
        del assign_value__1

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_3, full_7, float("0"), True)
        del cast_3

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_2 = paddle._C_ops.sqrt(scale_4)
        del scale_4

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_2 = paddle._C_ops.divide(transpose_6, sqrt_2)
        del sqrt_2, transpose_6

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x21x21xi64) <- (1x1x21x21xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(unsqueeze_9, full_11, float("0"), True)
        del full_11, unsqueeze_9

        # pd_op.scale: (1x1x21x21xi64) <- (1x1x21x21xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(scale_5, full_4, float("21"), True)
        del scale_5

        # pd_op.clip: (1x1x21x21xi64) <- (1x1x21x21xi64, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(scale_6, full_3, full_9)
        del full_9, scale_6

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_7 = paddle._C_ops.transpose(divide_2, [0, 1, 3, 2])
        del divide_2

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_5 = paddle._C_ops.matmul(split_1, transpose_7, False, False)
        del split_1, transpose_7

        # pd_op.expand: (1x12x21x21xi64) <- (1x1x21x21xi64, 4xi64)
        expand_4 = paddle._C_ops.expand(clip_1, full_int_array_13)
        del clip_1

        # pd_op.expand: (1x12x21x21xi64) <- (1x12x21x21xi64, 4xi64)
        expand_5 = paddle._C_ops.expand(expand_4, full_int_array_13)
        del expand_4, full_int_array_13

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_6 = paddle._C_ops.expand(matmul_5, full_int_array_14)
        del matmul_5

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_1 = paddle._C_ops.take_along_axis(expand_6, expand_5, 3)
        del expand_6

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_8 = paddle._C_ops.transpose(take_along_axis_1, [0, 1, 3, 2])
        del take_along_axis_1

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_5 = paddle._C_ops.add(scale_3, transpose_8)
        del scale_3, transpose_8

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_6 = paddle._C_ops.add(matmul_1, add_5)
        del add_5, matmul_1

        # pd_op.cast: (1x1x21x21xb) <- (1x1x21x21xf32)
        cast_4 = paddle._C_ops.cast(cast_1, paddle.bool)
        del cast_1

        # pd_op.logical_not: (1x1x21x21xb) <- (1x1x21x21xb)
        logical_not_0 = paddle._C_ops.logical_not(cast_4)
        del cast_4

        # pd_op.full: (1x12x21x21xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1, 12, 21, 21],
            float("-inf"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            full_12, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            add_6, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full_like: (1x1x21x21xb) <- (1x1x21x21xb, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            logical_not_0,
            full_3,
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (1x1x21x21xf32) <- (1x1x21x21xb)
        cast_5 = paddle._C_ops.cast(full_like_3, paddle.float32)
        del full_like_3

        # pd_op.cast: (1x1x21x21xf32) <- (1x1x21x21xb)
        cast_6 = paddle._C_ops.cast(logical_not_0, paddle.float32)
        del logical_not_0

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_7 = paddle._C_ops.add(full_like_1, full_like_2)
        del full_like_2

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_8 = paddle._C_ops.add(add_7, cast_5)
        del add_7

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_9 = paddle._C_ops.add(full_12, add_8)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_10 = paddle._C_ops.add(add_6, add_8)
        del add_6

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_11 = paddle._C_ops.add(cast_6, add_8)
        del add_8

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_7 = paddle._C_ops.cast(add_11, paddle.bool)
        del add_11

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_0 = paddle._C_ops.where(cast_7, add_9, add_10)
        del add_10, add_9, cast_7

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_0 = paddle._C_ops.softmax(where_0, -1)
        del where_0

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_6 = paddle._C_ops.matmul(softmax_0, add_3, False, False)
        del add_3, softmax_0

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])
        del matmul_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_15 = [1, 21, -1]

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_9, full_int_array_15)
        del transpose_9

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_5, parameter_186, False, False)
        del parameter_186, reshape_5

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_12 = paddle._C_ops.add(matmul_7, parameter_185)
        del matmul_7, parameter_185

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_13 = paddle._C_ops.add(add_12, multiply_1)
        del add_12

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_2 = paddle._C_ops.mean(add_13, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_2 = paddle._C_ops.subtract(add_13, mean_2)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_1 = paddle._C_ops.pow(subtract_2, float("2"))
        del subtract_2

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_3 = paddle._C_ops.mean(pow_1, full_int_array_2, True)
        del pow_1

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_3 = paddle._C_ops.subtract(add_13, mean_2)
        del add_13, mean_2

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(mean_3, full_4, float("1e-07"), True)
        del mean_3

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_3 = paddle._C_ops.sqrt(scale_7)
        del scale_7

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_3 = paddle._C_ops.divide(subtract_3, sqrt_3)
        del sqrt_3, subtract_3

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_3 = paddle._C_ops.multiply(parameter_184, divide_3)
        del divide_3, parameter_184

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_14 = paddle._C_ops.add(multiply_3, parameter_183)
        del multiply_3, parameter_183

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_8 = paddle._C_ops.matmul(add_14, parameter_182, False, False)
        del parameter_182

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_15 = paddle._C_ops.add(matmul_8, parameter_181)
        del matmul_8, parameter_181

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_0 = paddle._C_ops.gelu(add_15, False)
        del add_15

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_9 = paddle._C_ops.matmul(gelu_0, parameter_180, False, False)
        del gelu_0, parameter_180

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_16 = paddle._C_ops.add(matmul_9, parameter_179)
        del matmul_9, parameter_179

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_17 = paddle._C_ops.add(add_16, add_14)
        del add_14, add_16

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_4 = paddle._C_ops.mean(add_17, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_4 = paddle._C_ops.subtract(add_17, mean_4)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_2 = paddle._C_ops.pow(subtract_4, float("2"))
        del subtract_4

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_5 = paddle._C_ops.mean(pow_2, full_int_array_2, True)
        del pow_2

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_5 = paddle._C_ops.subtract(add_17, mean_4)
        del add_17, mean_4

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(mean_5, full_4, float("1e-07"), True)
        del mean_5

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_4 = paddle._C_ops.sqrt(scale_8)
        del scale_8

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_4 = paddle._C_ops.divide(subtract_5, sqrt_4)
        del sqrt_4, subtract_5

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_4 = paddle._C_ops.multiply(parameter_178, divide_4)
        del divide_4, parameter_178

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_18 = paddle._C_ops.add(multiply_4, parameter_177)
        del multiply_4, parameter_177

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_10 = paddle._C_ops.matmul(add_18, parameter_174, False, False)
        del parameter_174

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(matmul_10, full_int_array_7)
        del matmul_10

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(transpose_10, 3, full_5)
        del transpose_10

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_3,
            split_4,
            split_5,
        ) = split_with_num_1
        del split_with_num_1

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(parameter_176, full_int_array_8)
        del parameter_176

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(unsqueeze_11, full_int_array_9)
        del unsqueeze_11

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_19 = paddle._C_ops.add(split_3, transpose_11)
        del split_3, transpose_11

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(parameter_175, full_int_array_8)
        del parameter_175

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(unsqueeze_12, full_int_array_9)
        del unsqueeze_12

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_12 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_20 = paddle._C_ops.add(split_5, transpose_12)
        del split_5, transpose_12

        # pd_op.full: (xi64) <- ()
        full_13 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__2 = paddle._C_ops.assign_value_(
            full_13,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_13

        # pd_op.cast: (xf32) <- (xi64)
        cast_8 = paddle._C_ops.cast(assign_value__2, paddle.float32)
        del assign_value__2

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_8, full_7, float("0"), True)
        del cast_8

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_5 = paddle._C_ops.sqrt(scale_9)
        del scale_9

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_5 = paddle._C_ops.divide(add_19, sqrt_5)
        del add_19, sqrt_5

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_13 = paddle._C_ops.transpose(split_4, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_11 = paddle._C_ops.matmul(divide_5, transpose_13, False, False)
        del transpose_13

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            dropout_2, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_2

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(slice_1, full_int_array_0)
        del slice_1

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_12 = paddle._C_ops.matmul(unsqueeze_13, parameter_173, False, False)
        del parameter_173

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(matmul_12, full_int_array_12)
        del matmul_12

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_15 = paddle._C_ops.transpose(transpose_14, [0, 1, 3, 2])
        del transpose_14

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_13 = paddle._C_ops.matmul(divide_5, transpose_15, False, False)
        del divide_5, transpose_15

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_7 = paddle._C_ops.expand(matmul_13, full_int_array_14)
        del matmul_13

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_2 = paddle._C_ops.take_along_axis(expand_7, expand_2, 3)
        del expand_7

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(take_along_axis_2, full_4, float("0"), True)
        del take_along_axis_2

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_14 = paddle._C_ops.matmul(unsqueeze_13, parameter_172, False, False)
        del parameter_172, unsqueeze_13

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_21 = paddle._C_ops.add(matmul_14, parameter_171)
        del matmul_14, parameter_171

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_21, full_int_array_12)
        del add_21

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.full: (xi64) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__3 = paddle._C_ops.assign_value_(
            full_14,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_14

        # pd_op.cast: (xf32) <- (xi64)
        cast_9 = paddle._C_ops.cast(assign_value__3, paddle.float32)
        del assign_value__3

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cast_9, full_7, float("0"), True)
        del cast_9

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_6 = paddle._C_ops.sqrt(scale_11)
        del scale_11

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_6 = paddle._C_ops.divide(transpose_16, sqrt_6)
        del sqrt_6, transpose_16

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_17 = paddle._C_ops.transpose(divide_6, [0, 1, 3, 2])
        del divide_6

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_15 = paddle._C_ops.matmul(split_4, transpose_17, False, False)
        del split_4, transpose_17

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_8 = paddle._C_ops.expand(matmul_15, full_int_array_14)
        del matmul_15

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_3 = paddle._C_ops.take_along_axis(expand_8, expand_5, 3)
        del expand_8

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_18 = paddle._C_ops.transpose(take_along_axis_3, [0, 1, 3, 2])
        del take_along_axis_3

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_22 = paddle._C_ops.add(scale_10, transpose_18)
        del scale_10, transpose_18

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_23 = paddle._C_ops.add(matmul_11, add_22)
        del add_22, matmul_11

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            add_23, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_24 = paddle._C_ops.add(full_like_1, full_like_4)
        del full_like_4

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_25 = paddle._C_ops.add(add_24, cast_5)
        del add_24

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_26 = paddle._C_ops.add(full_12, add_25)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_27 = paddle._C_ops.add(add_23, add_25)
        del add_23

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_28 = paddle._C_ops.add(cast_6, add_25)
        del add_25

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_10 = paddle._C_ops.cast(add_28, paddle.bool)
        del add_28

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_1 = paddle._C_ops.where(cast_10, add_26, add_27)
        del add_26, add_27, cast_10

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_1 = paddle._C_ops.softmax(where_1, -1)
        del where_1

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_16 = paddle._C_ops.matmul(softmax_1, add_20, False, False)
        del add_20, softmax_1

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_19 = paddle._C_ops.transpose(matmul_16, [0, 2, 1, 3])
        del matmul_16

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_19, full_int_array_15)
        del transpose_19

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_17 = paddle._C_ops.matmul(reshape_11, parameter_170, False, False)
        del parameter_170, reshape_11

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_29 = paddle._C_ops.add(matmul_17, parameter_169)
        del matmul_17, parameter_169

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_30 = paddle._C_ops.add(add_29, add_18)
        del add_29

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_6 = paddle._C_ops.mean(add_30, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_6 = paddle._C_ops.subtract(add_30, mean_6)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_3 = paddle._C_ops.pow(subtract_6, float("2"))
        del subtract_6

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_7 = paddle._C_ops.mean(pow_3, full_int_array_2, True)
        del pow_3

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_7 = paddle._C_ops.subtract(add_30, mean_6)
        del add_30, mean_6

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(mean_7, full_4, float("1e-07"), True)
        del mean_7

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_7 = paddle._C_ops.sqrt(scale_12)
        del scale_12

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_7 = paddle._C_ops.divide(subtract_7, sqrt_7)
        del sqrt_7, subtract_7

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_5 = paddle._C_ops.multiply(parameter_168, divide_7)
        del divide_7, parameter_168

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_31 = paddle._C_ops.add(multiply_5, parameter_167)
        del multiply_5, parameter_167

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_18 = paddle._C_ops.matmul(add_31, parameter_166, False, False)
        del parameter_166

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_32 = paddle._C_ops.add(matmul_18, parameter_165)
        del matmul_18, parameter_165

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_1 = paddle._C_ops.gelu(add_32, False)
        del add_32

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_19 = paddle._C_ops.matmul(gelu_1, parameter_164, False, False)
        del gelu_1, parameter_164

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_33 = paddle._C_ops.add(matmul_19, parameter_163)
        del matmul_19, parameter_163

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_34 = paddle._C_ops.add(add_33, add_31)
        del add_31, add_33

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_8 = paddle._C_ops.mean(add_34, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_8 = paddle._C_ops.subtract(add_34, mean_8)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_4 = paddle._C_ops.pow(subtract_8, float("2"))
        del subtract_8

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_9 = paddle._C_ops.mean(pow_4, full_int_array_2, True)
        del pow_4

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_9 = paddle._C_ops.subtract(add_34, mean_8)
        del add_34, mean_8

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(mean_9, full_4, float("1e-07"), True)
        del mean_9

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_8 = paddle._C_ops.sqrt(scale_13)
        del scale_13

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_8 = paddle._C_ops.divide(subtract_9, sqrt_8)
        del sqrt_8, subtract_9

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_6 = paddle._C_ops.multiply(parameter_162, divide_8)
        del divide_8, parameter_162

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_35 = paddle._C_ops.add(multiply_6, parameter_161)
        del multiply_6, parameter_161

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_20 = paddle._C_ops.matmul(add_35, parameter_158, False, False)
        del parameter_158

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(matmul_20, full_int_array_7)
        del matmul_20

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_20 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(transpose_20, 3, full_5)
        del transpose_20

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_6,
            split_7,
            split_8,
        ) = split_with_num_2
        del split_with_num_2

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(parameter_160, full_int_array_8)
        del parameter_160

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(unsqueeze_14, full_int_array_9)
        del unsqueeze_14

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_36 = paddle._C_ops.add(split_6, transpose_21)
        del split_6, transpose_21

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(parameter_159, full_int_array_8)
        del parameter_159

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(unsqueeze_15, full_int_array_9)
        del unsqueeze_15

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_37 = paddle._C_ops.add(split_8, transpose_22)
        del split_8, transpose_22

        # pd_op.full: (xi64) <- ()
        full_15 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__4 = paddle._C_ops.assign_value_(
            full_15,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_15

        # pd_op.cast: (xf32) <- (xi64)
        cast_11 = paddle._C_ops.cast(assign_value__4, paddle.float32)
        del assign_value__4

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(cast_11, full_7, float("0"), True)
        del cast_11

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_9 = paddle._C_ops.sqrt(scale_14)
        del scale_14

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_9 = paddle._C_ops.divide(add_36, sqrt_9)
        del add_36, sqrt_9

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_23 = paddle._C_ops.transpose(split_7, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_21 = paddle._C_ops.matmul(divide_9, transpose_23, False, False)
        del transpose_23

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            dropout_4, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_4

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(slice_2, full_int_array_0)
        del slice_2

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_22 = paddle._C_ops.matmul(unsqueeze_16, parameter_157, False, False)
        del parameter_157

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(matmul_22, full_int_array_12)
        del matmul_22

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_24 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])
        del reshape_15

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_25 = paddle._C_ops.transpose(transpose_24, [0, 1, 3, 2])
        del transpose_24

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_23 = paddle._C_ops.matmul(divide_9, transpose_25, False, False)
        del divide_9, transpose_25

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_9 = paddle._C_ops.expand(matmul_23, full_int_array_14)
        del matmul_23

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_4 = paddle._C_ops.take_along_axis(expand_9, expand_2, 3)
        del expand_9

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(take_along_axis_4, full_4, float("0"), True)
        del take_along_axis_4

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_24 = paddle._C_ops.matmul(unsqueeze_16, parameter_156, False, False)
        del parameter_156, unsqueeze_16

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_38 = paddle._C_ops.add(matmul_24, parameter_155)
        del matmul_24, parameter_155

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_38, full_int_array_12)
        del add_38

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_26 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.full: (xi64) <- ()
        full_16 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__5 = paddle._C_ops.assign_value_(
            full_16,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_16

        # pd_op.cast: (xf32) <- (xi64)
        cast_12 = paddle._C_ops.cast(assign_value__5, paddle.float32)
        del assign_value__5

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_12, full_7, float("0"), True)
        del cast_12

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_10 = paddle._C_ops.sqrt(scale_16)
        del scale_16

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_10 = paddle._C_ops.divide(transpose_26, sqrt_10)
        del sqrt_10, transpose_26

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_27 = paddle._C_ops.transpose(divide_10, [0, 1, 3, 2])
        del divide_10

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_25 = paddle._C_ops.matmul(split_7, transpose_27, False, False)
        del split_7, transpose_27

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_10 = paddle._C_ops.expand(matmul_25, full_int_array_14)
        del matmul_25

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_5 = paddle._C_ops.take_along_axis(expand_10, expand_5, 3)
        del expand_10

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_28 = paddle._C_ops.transpose(take_along_axis_5, [0, 1, 3, 2])
        del take_along_axis_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_39 = paddle._C_ops.add(scale_15, transpose_28)
        del scale_15, transpose_28

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_40 = paddle._C_ops.add(matmul_21, add_39)
        del add_39, matmul_21

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            add_40, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_41 = paddle._C_ops.add(full_like_1, full_like_5)
        del full_like_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_42 = paddle._C_ops.add(add_41, cast_5)
        del add_41

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_43 = paddle._C_ops.add(full_12, add_42)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_44 = paddle._C_ops.add(add_40, add_42)
        del add_40

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_45 = paddle._C_ops.add(cast_6, add_42)
        del add_42

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_13 = paddle._C_ops.cast(add_45, paddle.bool)
        del add_45

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_2 = paddle._C_ops.where(cast_13, add_43, add_44)
        del add_43, add_44, cast_13

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_2 = paddle._C_ops.softmax(where_2, -1)
        del where_2

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_26 = paddle._C_ops.matmul(softmax_2, add_37, False, False)
        del add_37, softmax_2

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])
        del matmul_26

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_29, full_int_array_15)
        del transpose_29

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_27 = paddle._C_ops.matmul(reshape_17, parameter_154, False, False)
        del parameter_154, reshape_17

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_46 = paddle._C_ops.add(matmul_27, parameter_153)
        del matmul_27, parameter_153

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_47 = paddle._C_ops.add(add_46, add_35)
        del add_46

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_10 = paddle._C_ops.mean(add_47, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_10 = paddle._C_ops.subtract(add_47, mean_10)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_5 = paddle._C_ops.pow(subtract_10, float("2"))
        del subtract_10

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_11 = paddle._C_ops.mean(pow_5, full_int_array_2, True)
        del pow_5

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_11 = paddle._C_ops.subtract(add_47, mean_10)
        del add_47, mean_10

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(mean_11, full_4, float("1e-07"), True)
        del mean_11

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_11 = paddle._C_ops.sqrt(scale_17)
        del scale_17

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_11 = paddle._C_ops.divide(subtract_11, sqrt_11)
        del sqrt_11, subtract_11

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_7 = paddle._C_ops.multiply(parameter_152, divide_11)
        del divide_11, parameter_152

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_48 = paddle._C_ops.add(multiply_7, parameter_151)
        del multiply_7, parameter_151

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_28 = paddle._C_ops.matmul(add_48, parameter_150, False, False)
        del parameter_150

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_49 = paddle._C_ops.add(matmul_28, parameter_149)
        del matmul_28, parameter_149

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_2 = paddle._C_ops.gelu(add_49, False)
        del add_49

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_29 = paddle._C_ops.matmul(gelu_2, parameter_148, False, False)
        del gelu_2, parameter_148

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_50 = paddle._C_ops.add(matmul_29, parameter_147)
        del matmul_29, parameter_147

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_51 = paddle._C_ops.add(add_50, add_48)
        del add_48, add_50

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_12 = paddle._C_ops.mean(add_51, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_12 = paddle._C_ops.subtract(add_51, mean_12)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_6 = paddle._C_ops.pow(subtract_12, float("2"))
        del subtract_12

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_13 = paddle._C_ops.mean(pow_6, full_int_array_2, True)
        del pow_6

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_13 = paddle._C_ops.subtract(add_51, mean_12)
        del add_51, mean_12

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(mean_13, full_4, float("1e-07"), True)
        del mean_13

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_12 = paddle._C_ops.sqrt(scale_18)
        del scale_18

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_12 = paddle._C_ops.divide(subtract_13, sqrt_12)
        del sqrt_12, subtract_13

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_8 = paddle._C_ops.multiply(parameter_146, divide_12)
        del divide_12, parameter_146

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_52 = paddle._C_ops.add(multiply_8, parameter_145)
        del multiply_8, parameter_145

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_30 = paddle._C_ops.matmul(add_52, parameter_142, False, False)
        del parameter_142

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(matmul_30, full_int_array_7)
        del matmul_30

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(transpose_30, 3, full_5)
        del transpose_30

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_9,
            split_10,
            split_11,
        ) = split_with_num_3
        del split_with_num_3

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(parameter_144, full_int_array_8)
        del parameter_144

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(unsqueeze_17, full_int_array_9)
        del unsqueeze_17

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_53 = paddle._C_ops.add(split_9, transpose_31)
        del split_9, transpose_31

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(parameter_143, full_int_array_8)
        del parameter_143

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(unsqueeze_18, full_int_array_9)
        del unsqueeze_18

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_32 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_54 = paddle._C_ops.add(split_11, transpose_32)
        del split_11, transpose_32

        # pd_op.full: (xi64) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__6 = paddle._C_ops.assign_value_(
            full_17,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_17

        # pd_op.cast: (xf32) <- (xi64)
        cast_14 = paddle._C_ops.cast(assign_value__6, paddle.float32)
        del assign_value__6

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(cast_14, full_7, float("0"), True)
        del cast_14

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_13 = paddle._C_ops.sqrt(scale_19)
        del scale_19

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_13 = paddle._C_ops.divide(add_53, sqrt_13)
        del add_53, sqrt_13

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_33 = paddle._C_ops.transpose(split_10, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_31 = paddle._C_ops.matmul(divide_13, transpose_33, False, False)
        del transpose_33

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            dropout_6, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_6

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(slice_3, full_int_array_0)
        del slice_3

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_32 = paddle._C_ops.matmul(unsqueeze_19, parameter_141, False, False)
        del parameter_141

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(matmul_32, full_int_array_12)
        del matmul_32

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_34 = paddle._C_ops.transpose(reshape_21, [0, 2, 1, 3])
        del reshape_21

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_35 = paddle._C_ops.transpose(transpose_34, [0, 1, 3, 2])
        del transpose_34

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_33 = paddle._C_ops.matmul(divide_13, transpose_35, False, False)
        del divide_13, transpose_35

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_11 = paddle._C_ops.expand(matmul_33, full_int_array_14)
        del matmul_33

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_6 = paddle._C_ops.take_along_axis(expand_11, expand_2, 3)
        del expand_11

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(take_along_axis_6, full_4, float("0"), True)
        del take_along_axis_6

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_34 = paddle._C_ops.matmul(unsqueeze_19, parameter_140, False, False)
        del parameter_140, unsqueeze_19

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_55 = paddle._C_ops.add(matmul_34, parameter_139)
        del matmul_34, parameter_139

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_55, full_int_array_12)
        del add_55

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.full: (xi64) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__7 = paddle._C_ops.assign_value_(
            full_18,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_18

        # pd_op.cast: (xf32) <- (xi64)
        cast_15 = paddle._C_ops.cast(assign_value__7, paddle.float32)
        del assign_value__7

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(cast_15, full_7, float("0"), True)
        del cast_15

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_14 = paddle._C_ops.sqrt(scale_21)
        del scale_21

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_14 = paddle._C_ops.divide(transpose_36, sqrt_14)
        del sqrt_14, transpose_36

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_37 = paddle._C_ops.transpose(divide_14, [0, 1, 3, 2])
        del divide_14

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_35 = paddle._C_ops.matmul(split_10, transpose_37, False, False)
        del split_10, transpose_37

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_12 = paddle._C_ops.expand(matmul_35, full_int_array_14)
        del matmul_35

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_7 = paddle._C_ops.take_along_axis(expand_12, expand_5, 3)
        del expand_12

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_38 = paddle._C_ops.transpose(take_along_axis_7, [0, 1, 3, 2])
        del take_along_axis_7

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_56 = paddle._C_ops.add(scale_20, transpose_38)
        del scale_20, transpose_38

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_57 = paddle._C_ops.add(matmul_31, add_56)
        del add_56, matmul_31

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            add_57, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_58 = paddle._C_ops.add(full_like_1, full_like_6)
        del full_like_6

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_59 = paddle._C_ops.add(add_58, cast_5)
        del add_58

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_60 = paddle._C_ops.add(full_12, add_59)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_61 = paddle._C_ops.add(add_57, add_59)
        del add_57

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_62 = paddle._C_ops.add(cast_6, add_59)
        del add_59

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_16 = paddle._C_ops.cast(add_62, paddle.bool)
        del add_62

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_3 = paddle._C_ops.where(cast_16, add_60, add_61)
        del add_60, add_61, cast_16

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_3 = paddle._C_ops.softmax(where_3, -1)
        del where_3

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_36 = paddle._C_ops.matmul(softmax_3, add_54, False, False)
        del add_54, softmax_3

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_39 = paddle._C_ops.transpose(matmul_36, [0, 2, 1, 3])
        del matmul_36

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_23 = paddle._C_ops.reshape(transpose_39, full_int_array_15)
        del transpose_39

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_37 = paddle._C_ops.matmul(reshape_23, parameter_138, False, False)
        del parameter_138, reshape_23

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_63 = paddle._C_ops.add(matmul_37, parameter_137)
        del matmul_37, parameter_137

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_64 = paddle._C_ops.add(add_63, add_52)
        del add_63

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_14 = paddle._C_ops.mean(add_64, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_14 = paddle._C_ops.subtract(add_64, mean_14)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_7 = paddle._C_ops.pow(subtract_14, float("2"))
        del subtract_14

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_15 = paddle._C_ops.mean(pow_7, full_int_array_2, True)
        del pow_7

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_15 = paddle._C_ops.subtract(add_64, mean_14)
        del add_64, mean_14

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(mean_15, full_4, float("1e-07"), True)
        del mean_15

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_15 = paddle._C_ops.sqrt(scale_22)
        del scale_22

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_15 = paddle._C_ops.divide(subtract_15, sqrt_15)
        del sqrt_15, subtract_15

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_9 = paddle._C_ops.multiply(parameter_136, divide_15)
        del divide_15, parameter_136

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_65 = paddle._C_ops.add(multiply_9, parameter_135)
        del multiply_9, parameter_135

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_38 = paddle._C_ops.matmul(add_65, parameter_134, False, False)
        del parameter_134

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_66 = paddle._C_ops.add(matmul_38, parameter_133)
        del matmul_38, parameter_133

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_3 = paddle._C_ops.gelu(add_66, False)
        del add_66

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_39 = paddle._C_ops.matmul(gelu_3, parameter_132, False, False)
        del gelu_3, parameter_132

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_67 = paddle._C_ops.add(matmul_39, parameter_131)
        del matmul_39, parameter_131

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_68 = paddle._C_ops.add(add_67, add_65)
        del add_65, add_67

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_16 = paddle._C_ops.mean(add_68, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_16 = paddle._C_ops.subtract(add_68, mean_16)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_8 = paddle._C_ops.pow(subtract_16, float("2"))
        del subtract_16

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_17 = paddle._C_ops.mean(pow_8, full_int_array_2, True)
        del pow_8

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_17 = paddle._C_ops.subtract(add_68, mean_16)
        del add_68, mean_16

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(mean_17, full_4, float("1e-07"), True)
        del mean_17

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_16 = paddle._C_ops.sqrt(scale_23)
        del scale_23

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_16 = paddle._C_ops.divide(subtract_17, sqrt_16)
        del sqrt_16, subtract_17

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_10 = paddle._C_ops.multiply(parameter_130, divide_16)
        del divide_16, parameter_130

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_69 = paddle._C_ops.add(multiply_10, parameter_129)
        del multiply_10, parameter_129

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_40 = paddle._C_ops.matmul(add_69, parameter_126, False, False)
        del parameter_126

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(matmul_40, full_int_array_7)
        del matmul_40

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_40 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(transpose_40, 3, full_5)
        del transpose_40

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_12,
            split_13,
            split_14,
        ) = split_with_num_4
        del split_with_num_4

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(parameter_128, full_int_array_8)
        del parameter_128

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(unsqueeze_20, full_int_array_9)
        del unsqueeze_20

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_41 = paddle._C_ops.transpose(reshape_25, [0, 2, 1, 3])
        del reshape_25

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_70 = paddle._C_ops.add(split_12, transpose_41)
        del split_12, transpose_41

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(parameter_127, full_int_array_8)
        del parameter_127

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(unsqueeze_21, full_int_array_9)
        del unsqueeze_21

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_71 = paddle._C_ops.add(split_14, transpose_42)
        del split_14, transpose_42

        # pd_op.full: (xi64) <- ()
        full_19 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__8 = paddle._C_ops.assign_value_(
            full_19,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_19

        # pd_op.cast: (xf32) <- (xi64)
        cast_17 = paddle._C_ops.cast(assign_value__8, paddle.float32)
        del assign_value__8

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(cast_17, full_7, float("0"), True)
        del cast_17

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_17 = paddle._C_ops.sqrt(scale_24)
        del scale_24

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_17 = paddle._C_ops.divide(add_70, sqrt_17)
        del add_70, sqrt_17

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_43 = paddle._C_ops.transpose(split_13, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_41 = paddle._C_ops.matmul(divide_17, transpose_43, False, False)
        del transpose_43

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            dropout_8, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_8

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(slice_4, full_int_array_0)
        del slice_4

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_42 = paddle._C_ops.matmul(unsqueeze_22, parameter_125, False, False)
        del parameter_125

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(matmul_42, full_int_array_12)
        del matmul_42

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_45 = paddle._C_ops.transpose(transpose_44, [0, 1, 3, 2])
        del transpose_44

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_43 = paddle._C_ops.matmul(divide_17, transpose_45, False, False)
        del divide_17, transpose_45

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_13 = paddle._C_ops.expand(matmul_43, full_int_array_14)
        del matmul_43

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_8 = paddle._C_ops.take_along_axis(expand_13, expand_2, 3)
        del expand_13

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(take_along_axis_8, full_4, float("0"), True)
        del take_along_axis_8

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_44 = paddle._C_ops.matmul(unsqueeze_22, parameter_124, False, False)
        del parameter_124, unsqueeze_22

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_72 = paddle._C_ops.add(matmul_44, parameter_123)
        del matmul_44, parameter_123

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_72, full_int_array_12)
        del add_72

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.full: (xi64) <- ()
        full_20 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__9 = paddle._C_ops.assign_value_(
            full_20,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_20

        # pd_op.cast: (xf32) <- (xi64)
        cast_18 = paddle._C_ops.cast(assign_value__9, paddle.float32)
        del assign_value__9

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cast_18, full_7, float("0"), True)
        del cast_18

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_18 = paddle._C_ops.sqrt(scale_26)
        del scale_26

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_18 = paddle._C_ops.divide(transpose_46, sqrt_18)
        del sqrt_18, transpose_46

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_47 = paddle._C_ops.transpose(divide_18, [0, 1, 3, 2])
        del divide_18

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_45 = paddle._C_ops.matmul(split_13, transpose_47, False, False)
        del split_13, transpose_47

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_14 = paddle._C_ops.expand(matmul_45, full_int_array_14)
        del matmul_45

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_9 = paddle._C_ops.take_along_axis(expand_14, expand_5, 3)
        del expand_14

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_48 = paddle._C_ops.transpose(take_along_axis_9, [0, 1, 3, 2])
        del take_along_axis_9

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_73 = paddle._C_ops.add(scale_25, transpose_48)
        del scale_25, transpose_48

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_74 = paddle._C_ops.add(matmul_41, add_73)
        del add_73, matmul_41

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            add_74, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_75 = paddle._C_ops.add(full_like_1, full_like_7)
        del full_like_7

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_76 = paddle._C_ops.add(add_75, cast_5)
        del add_75

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_77 = paddle._C_ops.add(full_12, add_76)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_78 = paddle._C_ops.add(add_74, add_76)
        del add_74

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_79 = paddle._C_ops.add(cast_6, add_76)
        del add_76

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_19 = paddle._C_ops.cast(add_79, paddle.bool)
        del add_79

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_4 = paddle._C_ops.where(cast_19, add_77, add_78)
        del add_77, add_78, cast_19

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_4 = paddle._C_ops.softmax(where_4, -1)
        del where_4

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_46 = paddle._C_ops.matmul(softmax_4, add_71, False, False)
        del add_71, softmax_4

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_49 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_49, full_int_array_15)
        del transpose_49

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_29, parameter_122, False, False)
        del parameter_122, reshape_29

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_80 = paddle._C_ops.add(matmul_47, parameter_121)
        del matmul_47, parameter_121

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_81 = paddle._C_ops.add(add_80, add_69)
        del add_80

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_18 = paddle._C_ops.mean(add_81, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_18 = paddle._C_ops.subtract(add_81, mean_18)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_9 = paddle._C_ops.pow(subtract_18, float("2"))
        del subtract_18

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_19 = paddle._C_ops.mean(pow_9, full_int_array_2, True)
        del pow_9

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_19 = paddle._C_ops.subtract(add_81, mean_18)
        del add_81, mean_18

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(mean_19, full_4, float("1e-07"), True)
        del mean_19

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_19 = paddle._C_ops.sqrt(scale_27)
        del scale_27

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_19 = paddle._C_ops.divide(subtract_19, sqrt_19)
        del sqrt_19, subtract_19

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_11 = paddle._C_ops.multiply(parameter_120, divide_19)
        del divide_19, parameter_120

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_82 = paddle._C_ops.add(multiply_11, parameter_119)
        del multiply_11, parameter_119

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_48 = paddle._C_ops.matmul(add_82, parameter_118, False, False)
        del parameter_118

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_83 = paddle._C_ops.add(matmul_48, parameter_117)
        del matmul_48, parameter_117

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_4 = paddle._C_ops.gelu(add_83, False)
        del add_83

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_49 = paddle._C_ops.matmul(gelu_4, parameter_116, False, False)
        del gelu_4, parameter_116

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_84 = paddle._C_ops.add(matmul_49, parameter_115)
        del matmul_49, parameter_115

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_85 = paddle._C_ops.add(add_84, add_82)
        del add_82, add_84

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_20 = paddle._C_ops.mean(add_85, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_20 = paddle._C_ops.subtract(add_85, mean_20)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_10 = paddle._C_ops.pow(subtract_20, float("2"))
        del subtract_20

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_21 = paddle._C_ops.mean(pow_10, full_int_array_2, True)
        del pow_10

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_21 = paddle._C_ops.subtract(add_85, mean_20)
        del add_85, mean_20

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(mean_21, full_4, float("1e-07"), True)
        del mean_21

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_20 = paddle._C_ops.sqrt(scale_28)
        del scale_28

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_20 = paddle._C_ops.divide(subtract_21, sqrt_20)
        del sqrt_20, subtract_21

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_12 = paddle._C_ops.multiply(parameter_114, divide_20)
        del divide_20, parameter_114

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_86 = paddle._C_ops.add(multiply_12, parameter_113)
        del multiply_12, parameter_113

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_50 = paddle._C_ops.matmul(add_86, parameter_110, False, False)
        del parameter_110

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(matmul_50, full_int_array_7)
        del matmul_50

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(transpose_50, 3, full_5)
        del transpose_50

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_15,
            split_16,
            split_17,
        ) = split_with_num_5
        del split_with_num_5

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(parameter_112, full_int_array_8)
        del parameter_112

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(unsqueeze_23, full_int_array_9)
        del unsqueeze_23

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_31, [0, 2, 1, 3])
        del reshape_31

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_87 = paddle._C_ops.add(split_15, transpose_51)
        del split_15, transpose_51

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(parameter_111, full_int_array_8)
        del parameter_111

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(unsqueeze_24, full_int_array_9)
        del unsqueeze_24

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_52 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_88 = paddle._C_ops.add(split_17, transpose_52)
        del split_17, transpose_52

        # pd_op.full: (xi64) <- ()
        full_21 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__10 = paddle._C_ops.assign_value_(
            full_21,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_21

        # pd_op.cast: (xf32) <- (xi64)
        cast_20 = paddle._C_ops.cast(assign_value__10, paddle.float32)
        del assign_value__10

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(cast_20, full_7, float("0"), True)
        del cast_20

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_21 = paddle._C_ops.sqrt(scale_29)
        del scale_29

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_21 = paddle._C_ops.divide(add_87, sqrt_21)
        del add_87, sqrt_21

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_53 = paddle._C_ops.transpose(split_16, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_51 = paddle._C_ops.matmul(divide_21, transpose_53, False, False)
        del transpose_53

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            dropout_10, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_10

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(slice_5, full_int_array_0)
        del slice_5

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_52 = paddle._C_ops.matmul(unsqueeze_25, parameter_109, False, False)
        del parameter_109

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(matmul_52, full_int_array_12)
        del matmul_52

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_54 = paddle._C_ops.transpose(reshape_33, [0, 2, 1, 3])
        del reshape_33

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_55 = paddle._C_ops.transpose(transpose_54, [0, 1, 3, 2])
        del transpose_54

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_53 = paddle._C_ops.matmul(divide_21, transpose_55, False, False)
        del divide_21, transpose_55

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_15 = paddle._C_ops.expand(matmul_53, full_int_array_14)
        del matmul_53

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_10 = paddle._C_ops.take_along_axis(expand_15, expand_2, 3)
        del expand_15

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(take_along_axis_10, full_4, float("0"), True)
        del take_along_axis_10

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_54 = paddle._C_ops.matmul(unsqueeze_25, parameter_108, False, False)
        del parameter_108, unsqueeze_25

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_89 = paddle._C_ops.add(matmul_54, parameter_107)
        del matmul_54, parameter_107

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_89, full_int_array_12)
        del add_89

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.full: (xi64) <- ()
        full_22 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__11 = paddle._C_ops.assign_value_(
            full_22,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_22

        # pd_op.cast: (xf32) <- (xi64)
        cast_21 = paddle._C_ops.cast(assign_value__11, paddle.float32)
        del assign_value__11

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(cast_21, full_7, float("0"), True)
        del cast_21

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_22 = paddle._C_ops.sqrt(scale_31)
        del scale_31

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_22 = paddle._C_ops.divide(transpose_56, sqrt_22)
        del sqrt_22, transpose_56

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_57 = paddle._C_ops.transpose(divide_22, [0, 1, 3, 2])
        del divide_22

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_55 = paddle._C_ops.matmul(split_16, transpose_57, False, False)
        del split_16, transpose_57

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_16 = paddle._C_ops.expand(matmul_55, full_int_array_14)
        del matmul_55

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_11 = paddle._C_ops.take_along_axis(expand_16, expand_5, 3)
        del expand_16

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_58 = paddle._C_ops.transpose(take_along_axis_11, [0, 1, 3, 2])
        del take_along_axis_11

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_90 = paddle._C_ops.add(scale_30, transpose_58)
        del scale_30, transpose_58

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_91 = paddle._C_ops.add(matmul_51, add_90)
        del add_90, matmul_51

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_8 = paddle._C_ops.full_like(
            add_91, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_92 = paddle._C_ops.add(full_like_1, full_like_8)
        del full_like_8

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_93 = paddle._C_ops.add(add_92, cast_5)
        del add_92

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_94 = paddle._C_ops.add(full_12, add_93)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_95 = paddle._C_ops.add(add_91, add_93)
        del add_91

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_96 = paddle._C_ops.add(cast_6, add_93)
        del add_93

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_22 = paddle._C_ops.cast(add_96, paddle.bool)
        del add_96

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_5 = paddle._C_ops.where(cast_22, add_94, add_95)
        del add_94, add_95, cast_22

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_5 = paddle._C_ops.softmax(where_5, -1)
        del where_5

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_56 = paddle._C_ops.matmul(softmax_5, add_88, False, False)
        del add_88, softmax_5

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_59 = paddle._C_ops.transpose(matmul_56, [0, 2, 1, 3])
        del matmul_56

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_35 = paddle._C_ops.reshape(transpose_59, full_int_array_15)
        del transpose_59

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_57 = paddle._C_ops.matmul(reshape_35, parameter_106, False, False)
        del parameter_106, reshape_35

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_97 = paddle._C_ops.add(matmul_57, parameter_105)
        del matmul_57, parameter_105

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_98 = paddle._C_ops.add(add_97, add_86)
        del add_97

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_22 = paddle._C_ops.mean(add_98, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_22 = paddle._C_ops.subtract(add_98, mean_22)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_11 = paddle._C_ops.pow(subtract_22, float("2"))
        del subtract_22

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_23 = paddle._C_ops.mean(pow_11, full_int_array_2, True)
        del pow_11

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_23 = paddle._C_ops.subtract(add_98, mean_22)
        del add_98, mean_22

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(mean_23, full_4, float("1e-07"), True)
        del mean_23

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_23 = paddle._C_ops.sqrt(scale_32)
        del scale_32

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_23 = paddle._C_ops.divide(subtract_23, sqrt_23)
        del sqrt_23, subtract_23

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_13 = paddle._C_ops.multiply(parameter_104, divide_23)
        del divide_23, parameter_104

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_99 = paddle._C_ops.add(multiply_13, parameter_103)
        del multiply_13, parameter_103

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_58 = paddle._C_ops.matmul(add_99, parameter_102, False, False)
        del parameter_102

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_100 = paddle._C_ops.add(matmul_58, parameter_101)
        del matmul_58, parameter_101

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_5 = paddle._C_ops.gelu(add_100, False)
        del add_100

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_59 = paddle._C_ops.matmul(gelu_5, parameter_100, False, False)
        del gelu_5, parameter_100

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_101 = paddle._C_ops.add(matmul_59, parameter_99)
        del matmul_59, parameter_99

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_102 = paddle._C_ops.add(add_101, add_99)
        del add_101, add_99

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_24 = paddle._C_ops.mean(add_102, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_24 = paddle._C_ops.subtract(add_102, mean_24)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_12 = paddle._C_ops.pow(subtract_24, float("2"))
        del subtract_24

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_25 = paddle._C_ops.mean(pow_12, full_int_array_2, True)
        del pow_12

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_25 = paddle._C_ops.subtract(add_102, mean_24)
        del add_102, mean_24

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(mean_25, full_4, float("1e-07"), True)
        del mean_25

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_24 = paddle._C_ops.sqrt(scale_33)
        del scale_33

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_24 = paddle._C_ops.divide(subtract_25, sqrt_24)
        del sqrt_24, subtract_25

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_14 = paddle._C_ops.multiply(parameter_98, divide_24)
        del divide_24, parameter_98

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_103 = paddle._C_ops.add(multiply_14, parameter_97)
        del multiply_14, parameter_97

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_60 = paddle._C_ops.matmul(add_103, parameter_94, False, False)
        del parameter_94

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(matmul_60, full_int_array_7)
        del matmul_60

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_60 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(transpose_60, 3, full_5)
        del transpose_60

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_18,
            split_19,
            split_20,
        ) = split_with_num_6
        del split_with_num_6

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_26 = paddle._C_ops.unsqueeze(parameter_96, full_int_array_8)
        del parameter_96

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(unsqueeze_26, full_int_array_9)
        del unsqueeze_26

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_61 = paddle._C_ops.transpose(reshape_37, [0, 2, 1, 3])
        del reshape_37

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_104 = paddle._C_ops.add(split_18, transpose_61)
        del split_18, transpose_61

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_27 = paddle._C_ops.unsqueeze(parameter_95, full_int_array_8)
        del parameter_95

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(unsqueeze_27, full_int_array_9)
        del unsqueeze_27

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_105 = paddle._C_ops.add(split_20, transpose_62)
        del split_20, transpose_62

        # pd_op.full: (xi64) <- ()
        full_23 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__12 = paddle._C_ops.assign_value_(
            full_23,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_23

        # pd_op.cast: (xf32) <- (xi64)
        cast_23 = paddle._C_ops.cast(assign_value__12, paddle.float32)
        del assign_value__12

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(cast_23, full_7, float("0"), True)
        del cast_23

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_25 = paddle._C_ops.sqrt(scale_34)
        del scale_34

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_25 = paddle._C_ops.divide(add_104, sqrt_25)
        del add_104, sqrt_25

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_63 = paddle._C_ops.transpose(split_19, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_61 = paddle._C_ops.matmul(divide_25, transpose_63, False, False)
        del transpose_63

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            dropout_12, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_12

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_28 = paddle._C_ops.unsqueeze(slice_6, full_int_array_0)
        del slice_6

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_62 = paddle._C_ops.matmul(unsqueeze_28, parameter_93, False, False)
        del parameter_93

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(matmul_62, full_int_array_12)
        del matmul_62

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_39, [0, 2, 1, 3])
        del reshape_39

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_65 = paddle._C_ops.transpose(transpose_64, [0, 1, 3, 2])
        del transpose_64

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_63 = paddle._C_ops.matmul(divide_25, transpose_65, False, False)
        del divide_25, transpose_65

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_17 = paddle._C_ops.expand(matmul_63, full_int_array_14)
        del matmul_63

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_12 = paddle._C_ops.take_along_axis(expand_17, expand_2, 3)
        del expand_17

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(take_along_axis_12, full_4, float("0"), True)
        del take_along_axis_12

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_64 = paddle._C_ops.matmul(unsqueeze_28, parameter_92, False, False)
        del parameter_92, unsqueeze_28

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_106 = paddle._C_ops.add(matmul_64, parameter_91)
        del matmul_64, parameter_91

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_106, full_int_array_12)
        del add_106

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_66 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.full: (xi64) <- ()
        full_24 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__13 = paddle._C_ops.assign_value_(
            full_24,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_24

        # pd_op.cast: (xf32) <- (xi64)
        cast_24 = paddle._C_ops.cast(assign_value__13, paddle.float32)
        del assign_value__13

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(cast_24, full_7, float("0"), True)
        del cast_24

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_26 = paddle._C_ops.sqrt(scale_36)
        del scale_36

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_26 = paddle._C_ops.divide(transpose_66, sqrt_26)
        del sqrt_26, transpose_66

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_67 = paddle._C_ops.transpose(divide_26, [0, 1, 3, 2])
        del divide_26

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_65 = paddle._C_ops.matmul(split_19, transpose_67, False, False)
        del split_19, transpose_67

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_18 = paddle._C_ops.expand(matmul_65, full_int_array_14)
        del matmul_65

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_13 = paddle._C_ops.take_along_axis(expand_18, expand_5, 3)
        del expand_18

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_68 = paddle._C_ops.transpose(take_along_axis_13, [0, 1, 3, 2])
        del take_along_axis_13

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_107 = paddle._C_ops.add(scale_35, transpose_68)
        del scale_35, transpose_68

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_108 = paddle._C_ops.add(matmul_61, add_107)
        del add_107, matmul_61

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_9 = paddle._C_ops.full_like(
            add_108, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_109 = paddle._C_ops.add(full_like_1, full_like_9)
        del full_like_9

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_110 = paddle._C_ops.add(add_109, cast_5)
        del add_109

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_111 = paddle._C_ops.add(full_12, add_110)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_112 = paddle._C_ops.add(add_108, add_110)
        del add_108

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_113 = paddle._C_ops.add(cast_6, add_110)
        del add_110

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_25 = paddle._C_ops.cast(add_113, paddle.bool)
        del add_113

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_6 = paddle._C_ops.where(cast_25, add_111, add_112)
        del add_111, add_112, cast_25

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_6 = paddle._C_ops.softmax(where_6, -1)
        del where_6

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_66 = paddle._C_ops.matmul(softmax_6, add_105, False, False)
        del add_105, softmax_6

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])
        del matmul_66

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_69, full_int_array_15)
        del transpose_69

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_41, parameter_90, False, False)
        del parameter_90, reshape_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_114 = paddle._C_ops.add(matmul_67, parameter_89)
        del matmul_67, parameter_89

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_115 = paddle._C_ops.add(add_114, add_103)
        del add_114

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_26 = paddle._C_ops.mean(add_115, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_26 = paddle._C_ops.subtract(add_115, mean_26)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_13 = paddle._C_ops.pow(subtract_26, float("2"))
        del subtract_26

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_27 = paddle._C_ops.mean(pow_13, full_int_array_2, True)
        del pow_13

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_27 = paddle._C_ops.subtract(add_115, mean_26)
        del add_115, mean_26

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(mean_27, full_4, float("1e-07"), True)
        del mean_27

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_27 = paddle._C_ops.sqrt(scale_37)
        del scale_37

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_27 = paddle._C_ops.divide(subtract_27, sqrt_27)
        del sqrt_27, subtract_27

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_15 = paddle._C_ops.multiply(parameter_88, divide_27)
        del divide_27, parameter_88

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_116 = paddle._C_ops.add(multiply_15, parameter_87)
        del multiply_15, parameter_87

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_68 = paddle._C_ops.matmul(add_116, parameter_86, False, False)
        del parameter_86

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_117 = paddle._C_ops.add(matmul_68, parameter_85)
        del matmul_68, parameter_85

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_6 = paddle._C_ops.gelu(add_117, False)
        del add_117

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_69 = paddle._C_ops.matmul(gelu_6, parameter_84, False, False)
        del gelu_6, parameter_84

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_118 = paddle._C_ops.add(matmul_69, parameter_83)
        del matmul_69, parameter_83

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_119 = paddle._C_ops.add(add_118, add_116)
        del add_116, add_118

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_28 = paddle._C_ops.mean(add_119, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_28 = paddle._C_ops.subtract(add_119, mean_28)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_14 = paddle._C_ops.pow(subtract_28, float("2"))
        del subtract_28

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_29 = paddle._C_ops.mean(pow_14, full_int_array_2, True)
        del pow_14

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_29 = paddle._C_ops.subtract(add_119, mean_28)
        del add_119, mean_28

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(mean_29, full_4, float("1e-07"), True)
        del mean_29

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_28 = paddle._C_ops.sqrt(scale_38)
        del scale_38

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_28 = paddle._C_ops.divide(subtract_29, sqrt_28)
        del sqrt_28, subtract_29

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_16 = paddle._C_ops.multiply(parameter_82, divide_28)
        del divide_28, parameter_82

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_120 = paddle._C_ops.add(multiply_16, parameter_81)
        del multiply_16, parameter_81

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_70 = paddle._C_ops.matmul(add_120, parameter_78, False, False)
        del parameter_78

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(matmul_70, full_int_array_7)
        del matmul_70

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(transpose_70, 3, full_5)
        del transpose_70

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_21,
            split_22,
            split_23,
        ) = split_with_num_7
        del split_with_num_7

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_29 = paddle._C_ops.unsqueeze(parameter_80, full_int_array_8)
        del parameter_80

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(unsqueeze_29, full_int_array_9)
        del unsqueeze_29

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_43, [0, 2, 1, 3])
        del reshape_43

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_121 = paddle._C_ops.add(split_21, transpose_71)
        del split_21, transpose_71

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_30 = paddle._C_ops.unsqueeze(parameter_79, full_int_array_8)
        del parameter_79

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(unsqueeze_30, full_int_array_9)
        del unsqueeze_30

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_122 = paddle._C_ops.add(split_23, transpose_72)
        del split_23, transpose_72

        # pd_op.full: (xi64) <- ()
        full_25 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__14 = paddle._C_ops.assign_value_(
            full_25,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_25

        # pd_op.cast: (xf32) <- (xi64)
        cast_26 = paddle._C_ops.cast(assign_value__14, paddle.float32)
        del assign_value__14

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(cast_26, full_7, float("0"), True)
        del cast_26

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_29 = paddle._C_ops.sqrt(scale_39)
        del scale_39

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_29 = paddle._C_ops.divide(add_121, sqrt_29)
        del add_121, sqrt_29

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_73 = paddle._C_ops.transpose(split_22, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_71 = paddle._C_ops.matmul(divide_29, transpose_73, False, False)
        del transpose_73

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            dropout_14, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_14

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_31 = paddle._C_ops.unsqueeze(slice_7, full_int_array_0)
        del slice_7

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_72 = paddle._C_ops.matmul(unsqueeze_31, parameter_77, False, False)
        del parameter_77

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_45 = paddle._C_ops.reshape(matmul_72, full_int_array_12)
        del matmul_72

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_74 = paddle._C_ops.transpose(reshape_45, [0, 2, 1, 3])
        del reshape_45

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_75 = paddle._C_ops.transpose(transpose_74, [0, 1, 3, 2])
        del transpose_74

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_73 = paddle._C_ops.matmul(divide_29, transpose_75, False, False)
        del divide_29, transpose_75

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_19 = paddle._C_ops.expand(matmul_73, full_int_array_14)
        del matmul_73

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_14 = paddle._C_ops.take_along_axis(expand_19, expand_2, 3)
        del expand_19

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_40 = paddle._C_ops.scale(take_along_axis_14, full_4, float("0"), True)
        del take_along_axis_14

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_74 = paddle._C_ops.matmul(unsqueeze_31, parameter_76, False, False)
        del parameter_76, unsqueeze_31

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_123 = paddle._C_ops.add(matmul_74, parameter_75)
        del matmul_74, parameter_75

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_123, full_int_array_12)
        del add_123

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_76 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.full: (xi64) <- ()
        full_26 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__15 = paddle._C_ops.assign_value_(
            full_26,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_26

        # pd_op.cast: (xf32) <- (xi64)
        cast_27 = paddle._C_ops.cast(assign_value__15, paddle.float32)
        del assign_value__15

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_41 = paddle._C_ops.scale(cast_27, full_7, float("0"), True)
        del cast_27

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_30 = paddle._C_ops.sqrt(scale_41)
        del scale_41

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_30 = paddle._C_ops.divide(transpose_76, sqrt_30)
        del sqrt_30, transpose_76

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_77 = paddle._C_ops.transpose(divide_30, [0, 1, 3, 2])
        del divide_30

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_75 = paddle._C_ops.matmul(split_22, transpose_77, False, False)
        del split_22, transpose_77

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_20 = paddle._C_ops.expand(matmul_75, full_int_array_14)
        del matmul_75

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_15 = paddle._C_ops.take_along_axis(expand_20, expand_5, 3)
        del expand_20

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_78 = paddle._C_ops.transpose(take_along_axis_15, [0, 1, 3, 2])
        del take_along_axis_15

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_124 = paddle._C_ops.add(scale_40, transpose_78)
        del scale_40, transpose_78

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_125 = paddle._C_ops.add(matmul_71, add_124)
        del add_124, matmul_71

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_10 = paddle._C_ops.full_like(
            add_125, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_126 = paddle._C_ops.add(full_like_1, full_like_10)
        del full_like_10

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_127 = paddle._C_ops.add(add_126, cast_5)
        del add_126

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_128 = paddle._C_ops.add(full_12, add_127)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_129 = paddle._C_ops.add(add_125, add_127)
        del add_125

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_130 = paddle._C_ops.add(cast_6, add_127)
        del add_127

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_28 = paddle._C_ops.cast(add_130, paddle.bool)
        del add_130

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_7 = paddle._C_ops.where(cast_28, add_128, add_129)
        del add_128, add_129, cast_28

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_7 = paddle._C_ops.softmax(where_7, -1)
        del where_7

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_76 = paddle._C_ops.matmul(softmax_7, add_122, False, False)
        del add_122, softmax_7

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_79 = paddle._C_ops.transpose(matmul_76, [0, 2, 1, 3])
        del matmul_76

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_47 = paddle._C_ops.reshape(transpose_79, full_int_array_15)
        del transpose_79

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_77 = paddle._C_ops.matmul(reshape_47, parameter_74, False, False)
        del parameter_74, reshape_47

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_131 = paddle._C_ops.add(matmul_77, parameter_73)
        del matmul_77, parameter_73

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_132 = paddle._C_ops.add(add_131, add_120)
        del add_131

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_30 = paddle._C_ops.mean(add_132, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_30 = paddle._C_ops.subtract(add_132, mean_30)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_15 = paddle._C_ops.pow(subtract_30, float("2"))
        del subtract_30

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_31 = paddle._C_ops.mean(pow_15, full_int_array_2, True)
        del pow_15

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_31 = paddle._C_ops.subtract(add_132, mean_30)
        del add_132, mean_30

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_42 = paddle._C_ops.scale(mean_31, full_4, float("1e-07"), True)
        del mean_31

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_31 = paddle._C_ops.sqrt(scale_42)
        del scale_42

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_31 = paddle._C_ops.divide(subtract_31, sqrt_31)
        del sqrt_31, subtract_31

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_17 = paddle._C_ops.multiply(parameter_72, divide_31)
        del divide_31, parameter_72

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_133 = paddle._C_ops.add(multiply_17, parameter_71)
        del multiply_17, parameter_71

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_78 = paddle._C_ops.matmul(add_133, parameter_70, False, False)
        del parameter_70

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_134 = paddle._C_ops.add(matmul_78, parameter_69)
        del matmul_78, parameter_69

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_7 = paddle._C_ops.gelu(add_134, False)
        del add_134

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_79 = paddle._C_ops.matmul(gelu_7, parameter_68, False, False)
        del gelu_7, parameter_68

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_135 = paddle._C_ops.add(matmul_79, parameter_67)
        del matmul_79, parameter_67

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_136 = paddle._C_ops.add(add_135, add_133)
        del add_133, add_135

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_32 = paddle._C_ops.mean(add_136, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_32 = paddle._C_ops.subtract(add_136, mean_32)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_16 = paddle._C_ops.pow(subtract_32, float("2"))
        del subtract_32

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_33 = paddle._C_ops.mean(pow_16, full_int_array_2, True)
        del pow_16

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_33 = paddle._C_ops.subtract(add_136, mean_32)
        del add_136, mean_32

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_43 = paddle._C_ops.scale(mean_33, full_4, float("1e-07"), True)
        del mean_33

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_32 = paddle._C_ops.sqrt(scale_43)
        del scale_43

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_32 = paddle._C_ops.divide(subtract_33, sqrt_32)
        del sqrt_32, subtract_33

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_18 = paddle._C_ops.multiply(parameter_66, divide_32)
        del divide_32, parameter_66

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_137 = paddle._C_ops.add(multiply_18, parameter_65)
        del multiply_18, parameter_65

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_80 = paddle._C_ops.matmul(add_137, parameter_62, False, False)
        del parameter_62

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(matmul_80, full_int_array_7)
        del matmul_80

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_80 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(transpose_80, 3, full_5)
        del transpose_80

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_24,
            split_25,
            split_26,
        ) = split_with_num_8
        del split_with_num_8

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_32 = paddle._C_ops.unsqueeze(parameter_64, full_int_array_8)
        del parameter_64

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_49 = paddle._C_ops.reshape(unsqueeze_32, full_int_array_9)
        del unsqueeze_32

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_81 = paddle._C_ops.transpose(reshape_49, [0, 2, 1, 3])
        del reshape_49

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_138 = paddle._C_ops.add(split_24, transpose_81)
        del split_24, transpose_81

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_33 = paddle._C_ops.unsqueeze(parameter_63, full_int_array_8)
        del parameter_63

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(unsqueeze_33, full_int_array_9)
        del unsqueeze_33

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_82 = paddle._C_ops.transpose(reshape_50, [0, 2, 1, 3])
        del reshape_50

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_139 = paddle._C_ops.add(split_26, transpose_82)
        del split_26, transpose_82

        # pd_op.full: (xi64) <- ()
        full_27 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__16 = paddle._C_ops.assign_value_(
            full_27,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_27

        # pd_op.cast: (xf32) <- (xi64)
        cast_29 = paddle._C_ops.cast(assign_value__16, paddle.float32)
        del assign_value__16

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_44 = paddle._C_ops.scale(cast_29, full_7, float("0"), True)
        del cast_29

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_33 = paddle._C_ops.sqrt(scale_44)
        del scale_44

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_33 = paddle._C_ops.divide(add_138, sqrt_33)
        del add_138, sqrt_33

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_83 = paddle._C_ops.transpose(split_25, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_81 = paddle._C_ops.matmul(divide_33, transpose_83, False, False)
        del transpose_83

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            dropout_16, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_16

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_34 = paddle._C_ops.unsqueeze(slice_8, full_int_array_0)
        del slice_8

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_82 = paddle._C_ops.matmul(unsqueeze_34, parameter_61, False, False)
        del parameter_61

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_51 = paddle._C_ops.reshape(matmul_82, full_int_array_12)
        del matmul_82

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_84 = paddle._C_ops.transpose(reshape_51, [0, 2, 1, 3])
        del reshape_51

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_85 = paddle._C_ops.transpose(transpose_84, [0, 1, 3, 2])
        del transpose_84

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_83 = paddle._C_ops.matmul(divide_33, transpose_85, False, False)
        del divide_33, transpose_85

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_21 = paddle._C_ops.expand(matmul_83, full_int_array_14)
        del matmul_83

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_16 = paddle._C_ops.take_along_axis(expand_21, expand_2, 3)
        del expand_21

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_45 = paddle._C_ops.scale(take_along_axis_16, full_4, float("0"), True)
        del take_along_axis_16

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_84 = paddle._C_ops.matmul(unsqueeze_34, parameter_60, False, False)
        del parameter_60, unsqueeze_34

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_140 = paddle._C_ops.add(matmul_84, parameter_59)
        del matmul_84, parameter_59

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_52 = paddle._C_ops.reshape(add_140, full_int_array_12)
        del add_140

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_86 = paddle._C_ops.transpose(reshape_52, [0, 2, 1, 3])
        del reshape_52

        # pd_op.full: (xi64) <- ()
        full_28 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__17 = paddle._C_ops.assign_value_(
            full_28,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_28

        # pd_op.cast: (xf32) <- (xi64)
        cast_30 = paddle._C_ops.cast(assign_value__17, paddle.float32)
        del assign_value__17

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_46 = paddle._C_ops.scale(cast_30, full_7, float("0"), True)
        del cast_30

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_34 = paddle._C_ops.sqrt(scale_46)
        del scale_46

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_34 = paddle._C_ops.divide(transpose_86, sqrt_34)
        del sqrt_34, transpose_86

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_87 = paddle._C_ops.transpose(divide_34, [0, 1, 3, 2])
        del divide_34

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_85 = paddle._C_ops.matmul(split_25, transpose_87, False, False)
        del split_25, transpose_87

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_22 = paddle._C_ops.expand(matmul_85, full_int_array_14)
        del matmul_85

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_17 = paddle._C_ops.take_along_axis(expand_22, expand_5, 3)
        del expand_22

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_88 = paddle._C_ops.transpose(take_along_axis_17, [0, 1, 3, 2])
        del take_along_axis_17

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_141 = paddle._C_ops.add(scale_45, transpose_88)
        del scale_45, transpose_88

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_142 = paddle._C_ops.add(matmul_81, add_141)
        del add_141, matmul_81

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_11 = paddle._C_ops.full_like(
            add_142, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_143 = paddle._C_ops.add(full_like_1, full_like_11)
        del full_like_11

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_144 = paddle._C_ops.add(add_143, cast_5)
        del add_143

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_145 = paddle._C_ops.add(full_12, add_144)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_146 = paddle._C_ops.add(add_142, add_144)
        del add_142

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_147 = paddle._C_ops.add(cast_6, add_144)
        del add_144

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_31 = paddle._C_ops.cast(add_147, paddle.bool)
        del add_147

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_8 = paddle._C_ops.where(cast_31, add_145, add_146)
        del add_145, add_146, cast_31

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_8 = paddle._C_ops.softmax(where_8, -1)
        del where_8

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_86 = paddle._C_ops.matmul(softmax_8, add_139, False, False)
        del add_139, softmax_8

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_89 = paddle._C_ops.transpose(matmul_86, [0, 2, 1, 3])
        del matmul_86

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_53 = paddle._C_ops.reshape(transpose_89, full_int_array_15)
        del transpose_89

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_87 = paddle._C_ops.matmul(reshape_53, parameter_58, False, False)
        del parameter_58, reshape_53

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_148 = paddle._C_ops.add(matmul_87, parameter_57)
        del matmul_87, parameter_57

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_149 = paddle._C_ops.add(add_148, add_137)
        del add_148

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_34 = paddle._C_ops.mean(add_149, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_34 = paddle._C_ops.subtract(add_149, mean_34)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_17 = paddle._C_ops.pow(subtract_34, float("2"))
        del subtract_34

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_35 = paddle._C_ops.mean(pow_17, full_int_array_2, True)
        del pow_17

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_35 = paddle._C_ops.subtract(add_149, mean_34)
        del add_149, mean_34

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_47 = paddle._C_ops.scale(mean_35, full_4, float("1e-07"), True)
        del mean_35

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_35 = paddle._C_ops.sqrt(scale_47)
        del scale_47

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_35 = paddle._C_ops.divide(subtract_35, sqrt_35)
        del sqrt_35, subtract_35

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_19 = paddle._C_ops.multiply(parameter_56, divide_35)
        del divide_35, parameter_56

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_150 = paddle._C_ops.add(multiply_19, parameter_55)
        del multiply_19, parameter_55

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_88 = paddle._C_ops.matmul(add_150, parameter_54, False, False)
        del parameter_54

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_151 = paddle._C_ops.add(matmul_88, parameter_53)
        del matmul_88, parameter_53

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_8 = paddle._C_ops.gelu(add_151, False)
        del add_151

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_89 = paddle._C_ops.matmul(gelu_8, parameter_52, False, False)
        del gelu_8, parameter_52

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_152 = paddle._C_ops.add(matmul_89, parameter_51)
        del matmul_89, parameter_51

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_153 = paddle._C_ops.add(add_152, add_150)
        del add_150, add_152

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_36 = paddle._C_ops.mean(add_153, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_36 = paddle._C_ops.subtract(add_153, mean_36)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_18 = paddle._C_ops.pow(subtract_36, float("2"))
        del subtract_36

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_37 = paddle._C_ops.mean(pow_18, full_int_array_2, True)
        del pow_18

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_37 = paddle._C_ops.subtract(add_153, mean_36)
        del add_153, mean_36

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_48 = paddle._C_ops.scale(mean_37, full_4, float("1e-07"), True)
        del mean_37

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_36 = paddle._C_ops.sqrt(scale_48)
        del scale_48

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_36 = paddle._C_ops.divide(subtract_37, sqrt_36)
        del sqrt_36, subtract_37

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_20 = paddle._C_ops.multiply(parameter_50, divide_36)
        del divide_36, parameter_50

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_154 = paddle._C_ops.add(multiply_20, parameter_49)
        del multiply_20, parameter_49

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_90 = paddle._C_ops.matmul(add_154, parameter_46, False, False)
        del parameter_46

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_54 = paddle._C_ops.reshape(matmul_90, full_int_array_7)
        del matmul_90

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_90 = paddle._C_ops.transpose(reshape_54, [0, 2, 1, 3])
        del reshape_54

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(transpose_90, 3, full_5)
        del transpose_90

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_27,
            split_28,
            split_29,
        ) = split_with_num_9
        del split_with_num_9

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_35 = paddle._C_ops.unsqueeze(parameter_48, full_int_array_8)
        del parameter_48

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_55 = paddle._C_ops.reshape(unsqueeze_35, full_int_array_9)
        del unsqueeze_35

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_91 = paddle._C_ops.transpose(reshape_55, [0, 2, 1, 3])
        del reshape_55

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_155 = paddle._C_ops.add(split_27, transpose_91)
        del split_27, transpose_91

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_36 = paddle._C_ops.unsqueeze(parameter_47, full_int_array_8)
        del parameter_47

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_56 = paddle._C_ops.reshape(unsqueeze_36, full_int_array_9)
        del unsqueeze_36

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_92 = paddle._C_ops.transpose(reshape_56, [0, 2, 1, 3])
        del reshape_56

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_156 = paddle._C_ops.add(split_29, transpose_92)
        del split_29, transpose_92

        # pd_op.full: (xi64) <- ()
        full_29 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__18 = paddle._C_ops.assign_value_(
            full_29,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_29

        # pd_op.cast: (xf32) <- (xi64)
        cast_32 = paddle._C_ops.cast(assign_value__18, paddle.float32)
        del assign_value__18

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_49 = paddle._C_ops.scale(cast_32, full_7, float("0"), True)
        del cast_32

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_37 = paddle._C_ops.sqrt(scale_49)
        del scale_49

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_37 = paddle._C_ops.divide(add_155, sqrt_37)
        del add_155, sqrt_37

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_93 = paddle._C_ops.transpose(split_28, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_91 = paddle._C_ops.matmul(divide_37, transpose_93, False, False)
        del transpose_93

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            dropout_18, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_18

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_37 = paddle._C_ops.unsqueeze(slice_9, full_int_array_0)
        del slice_9

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_92 = paddle._C_ops.matmul(unsqueeze_37, parameter_45, False, False)
        del parameter_45

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_57 = paddle._C_ops.reshape(matmul_92, full_int_array_12)
        del matmul_92

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_94 = paddle._C_ops.transpose(reshape_57, [0, 2, 1, 3])
        del reshape_57

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_95 = paddle._C_ops.transpose(transpose_94, [0, 1, 3, 2])
        del transpose_94

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_93 = paddle._C_ops.matmul(divide_37, transpose_95, False, False)
        del divide_37, transpose_95

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_23 = paddle._C_ops.expand(matmul_93, full_int_array_14)
        del matmul_93

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_18 = paddle._C_ops.take_along_axis(expand_23, expand_2, 3)
        del expand_23

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_50 = paddle._C_ops.scale(take_along_axis_18, full_4, float("0"), True)
        del take_along_axis_18

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_94 = paddle._C_ops.matmul(unsqueeze_37, parameter_44, False, False)
        del parameter_44, unsqueeze_37

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_157 = paddle._C_ops.add(matmul_94, parameter_43)
        del matmul_94, parameter_43

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_58 = paddle._C_ops.reshape(add_157, full_int_array_12)
        del add_157

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_96 = paddle._C_ops.transpose(reshape_58, [0, 2, 1, 3])
        del reshape_58

        # pd_op.full: (xi64) <- ()
        full_30 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__19 = paddle._C_ops.assign_value_(
            full_30,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_30

        # pd_op.cast: (xf32) <- (xi64)
        cast_33 = paddle._C_ops.cast(assign_value__19, paddle.float32)
        del assign_value__19

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_51 = paddle._C_ops.scale(cast_33, full_7, float("0"), True)
        del cast_33

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_38 = paddle._C_ops.sqrt(scale_51)
        del scale_51

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_38 = paddle._C_ops.divide(transpose_96, sqrt_38)
        del sqrt_38, transpose_96

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_97 = paddle._C_ops.transpose(divide_38, [0, 1, 3, 2])
        del divide_38

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_95 = paddle._C_ops.matmul(split_28, transpose_97, False, False)
        del split_28, transpose_97

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_24 = paddle._C_ops.expand(matmul_95, full_int_array_14)
        del matmul_95

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_19 = paddle._C_ops.take_along_axis(expand_24, expand_5, 3)
        del expand_24

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_98 = paddle._C_ops.transpose(take_along_axis_19, [0, 1, 3, 2])
        del take_along_axis_19

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_158 = paddle._C_ops.add(scale_50, transpose_98)
        del scale_50, transpose_98

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_159 = paddle._C_ops.add(matmul_91, add_158)
        del add_158, matmul_91

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_12 = paddle._C_ops.full_like(
            add_159, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_160 = paddle._C_ops.add(full_like_1, full_like_12)
        del full_like_12

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_161 = paddle._C_ops.add(add_160, cast_5)
        del add_160

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_162 = paddle._C_ops.add(full_12, add_161)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_163 = paddle._C_ops.add(add_159, add_161)
        del add_159

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_164 = paddle._C_ops.add(cast_6, add_161)
        del add_161

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_34 = paddle._C_ops.cast(add_164, paddle.bool)
        del add_164

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_9 = paddle._C_ops.where(cast_34, add_162, add_163)
        del add_162, add_163, cast_34

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_9 = paddle._C_ops.softmax(where_9, -1)
        del where_9

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_96 = paddle._C_ops.matmul(softmax_9, add_156, False, False)
        del add_156, softmax_9

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_99 = paddle._C_ops.transpose(matmul_96, [0, 2, 1, 3])
        del matmul_96

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_59 = paddle._C_ops.reshape(transpose_99, full_int_array_15)
        del transpose_99

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_97 = paddle._C_ops.matmul(reshape_59, parameter_42, False, False)
        del parameter_42, reshape_59

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_165 = paddle._C_ops.add(matmul_97, parameter_41)
        del matmul_97, parameter_41

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_166 = paddle._C_ops.add(add_165, add_154)
        del add_165

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_38 = paddle._C_ops.mean(add_166, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_38 = paddle._C_ops.subtract(add_166, mean_38)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_19 = paddle._C_ops.pow(subtract_38, float("2"))
        del subtract_38

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_39 = paddle._C_ops.mean(pow_19, full_int_array_2, True)
        del pow_19

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_39 = paddle._C_ops.subtract(add_166, mean_38)
        del add_166, mean_38

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_52 = paddle._C_ops.scale(mean_39, full_4, float("1e-07"), True)
        del mean_39

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_39 = paddle._C_ops.sqrt(scale_52)
        del scale_52

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_39 = paddle._C_ops.divide(subtract_39, sqrt_39)
        del sqrt_39, subtract_39

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_21 = paddle._C_ops.multiply(parameter_40, divide_39)
        del divide_39, parameter_40

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_167 = paddle._C_ops.add(multiply_21, parameter_39)
        del multiply_21, parameter_39

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_98 = paddle._C_ops.matmul(add_167, parameter_38, False, False)
        del parameter_38

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_168 = paddle._C_ops.add(matmul_98, parameter_37)
        del matmul_98, parameter_37

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_9 = paddle._C_ops.gelu(add_168, False)
        del add_168

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_99 = paddle._C_ops.matmul(gelu_9, parameter_36, False, False)
        del gelu_9, parameter_36

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_169 = paddle._C_ops.add(matmul_99, parameter_35)
        del matmul_99, parameter_35

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_170 = paddle._C_ops.add(add_169, add_167)
        del add_167, add_169

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_40 = paddle._C_ops.mean(add_170, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_40 = paddle._C_ops.subtract(add_170, mean_40)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_20 = paddle._C_ops.pow(subtract_40, float("2"))
        del subtract_40

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_41 = paddle._C_ops.mean(pow_20, full_int_array_2, True)
        del pow_20

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_41 = paddle._C_ops.subtract(add_170, mean_40)
        del add_170, mean_40

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_53 = paddle._C_ops.scale(mean_41, full_4, float("1e-07"), True)
        del mean_41

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_40 = paddle._C_ops.sqrt(scale_53)
        del scale_53

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_40 = paddle._C_ops.divide(subtract_41, sqrt_40)
        del sqrt_40, subtract_41

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_22 = paddle._C_ops.multiply(parameter_34, divide_40)
        del divide_40, parameter_34

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_171 = paddle._C_ops.add(multiply_22, parameter_33)
        del multiply_22, parameter_33

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_100 = paddle._C_ops.matmul(add_171, parameter_30, False, False)
        del parameter_30

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_60 = paddle._C_ops.reshape(matmul_100, full_int_array_7)
        del matmul_100

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_100 = paddle._C_ops.transpose(reshape_60, [0, 2, 1, 3])
        del reshape_60

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(transpose_100, 3, full_5)
        del transpose_100

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_30,
            split_31,
            split_32,
        ) = split_with_num_10
        del split_with_num_10

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_38 = paddle._C_ops.unsqueeze(parameter_32, full_int_array_8)
        del parameter_32

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_61 = paddle._C_ops.reshape(unsqueeze_38, full_int_array_9)
        del unsqueeze_38

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_101 = paddle._C_ops.transpose(reshape_61, [0, 2, 1, 3])
        del reshape_61

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_172 = paddle._C_ops.add(split_30, transpose_101)
        del split_30, transpose_101

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_39 = paddle._C_ops.unsqueeze(parameter_31, full_int_array_8)
        del parameter_31

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_62 = paddle._C_ops.reshape(unsqueeze_39, full_int_array_9)
        del unsqueeze_39

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_102 = paddle._C_ops.transpose(reshape_62, [0, 2, 1, 3])
        del reshape_62

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_173 = paddle._C_ops.add(split_32, transpose_102)
        del split_32, transpose_102

        # pd_op.full: (xi64) <- ()
        full_31 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__20 = paddle._C_ops.assign_value_(
            full_31,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_31

        # pd_op.cast: (xf32) <- (xi64)
        cast_35 = paddle._C_ops.cast(assign_value__20, paddle.float32)
        del assign_value__20

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_54 = paddle._C_ops.scale(cast_35, full_7, float("0"), True)
        del cast_35

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_41 = paddle._C_ops.sqrt(scale_54)
        del scale_54

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_41 = paddle._C_ops.divide(add_172, sqrt_41)
        del add_172, sqrt_41

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_103 = paddle._C_ops.transpose(split_31, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_101 = paddle._C_ops.matmul(divide_41, transpose_103, False, False)
        del transpose_103

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            dropout_20, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_20

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_40 = paddle._C_ops.unsqueeze(slice_10, full_int_array_0)
        del slice_10

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_102 = paddle._C_ops.matmul(unsqueeze_40, parameter_29, False, False)
        del parameter_29

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_63 = paddle._C_ops.reshape(matmul_102, full_int_array_12)
        del matmul_102

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_104 = paddle._C_ops.transpose(reshape_63, [0, 2, 1, 3])
        del reshape_63

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_105 = paddle._C_ops.transpose(transpose_104, [0, 1, 3, 2])
        del transpose_104

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_103 = paddle._C_ops.matmul(divide_41, transpose_105, False, False)
        del divide_41, transpose_105

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_25 = paddle._C_ops.expand(matmul_103, full_int_array_14)
        del matmul_103

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_20 = paddle._C_ops.take_along_axis(expand_25, expand_2, 3)
        del expand_25

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_55 = paddle._C_ops.scale(take_along_axis_20, full_4, float("0"), True)
        del take_along_axis_20

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_104 = paddle._C_ops.matmul(unsqueeze_40, parameter_28, False, False)
        del parameter_28, unsqueeze_40

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_174 = paddle._C_ops.add(matmul_104, parameter_27)
        del matmul_104, parameter_27

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_64 = paddle._C_ops.reshape(add_174, full_int_array_12)
        del add_174

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_106 = paddle._C_ops.transpose(reshape_64, [0, 2, 1, 3])
        del reshape_64

        # pd_op.full: (xi64) <- ()
        full_32 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__21 = paddle._C_ops.assign_value_(
            full_32,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_32

        # pd_op.cast: (xf32) <- (xi64)
        cast_36 = paddle._C_ops.cast(assign_value__21, paddle.float32)
        del assign_value__21

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_56 = paddle._C_ops.scale(cast_36, full_7, float("0"), True)
        del cast_36

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_42 = paddle._C_ops.sqrt(scale_56)
        del scale_56

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_42 = paddle._C_ops.divide(transpose_106, sqrt_42)
        del sqrt_42, transpose_106

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_107 = paddle._C_ops.transpose(divide_42, [0, 1, 3, 2])
        del divide_42

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_105 = paddle._C_ops.matmul(split_31, transpose_107, False, False)
        del split_31, transpose_107

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_26 = paddle._C_ops.expand(matmul_105, full_int_array_14)
        del matmul_105

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_21 = paddle._C_ops.take_along_axis(expand_26, expand_5, 3)
        del expand_26

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_108 = paddle._C_ops.transpose(take_along_axis_21, [0, 1, 3, 2])
        del take_along_axis_21

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_175 = paddle._C_ops.add(scale_55, transpose_108)
        del scale_55, transpose_108

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_176 = paddle._C_ops.add(matmul_101, add_175)
        del add_175, matmul_101

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_13 = paddle._C_ops.full_like(
            add_176, full_3, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_177 = paddle._C_ops.add(full_like_1, full_like_13)
        del full_like_13

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_178 = paddle._C_ops.add(add_177, cast_5)
        del add_177

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_179 = paddle._C_ops.add(full_12, add_178)

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_180 = paddle._C_ops.add(add_176, add_178)
        del add_176

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_181 = paddle._C_ops.add(cast_6, add_178)
        del add_178

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_37 = paddle._C_ops.cast(add_181, paddle.bool)
        del add_181

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_10 = paddle._C_ops.where(cast_37, add_179, add_180)
        del add_179, add_180, cast_37

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_10 = paddle._C_ops.softmax(where_10, -1)
        del where_10

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_106 = paddle._C_ops.matmul(softmax_10, add_173, False, False)
        del add_173, softmax_10

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_109 = paddle._C_ops.transpose(matmul_106, [0, 2, 1, 3])
        del matmul_106

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_65 = paddle._C_ops.reshape(transpose_109, full_int_array_15)
        del transpose_109

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_107 = paddle._C_ops.matmul(reshape_65, parameter_26, False, False)
        del parameter_26, reshape_65

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_182 = paddle._C_ops.add(matmul_107, parameter_25)
        del matmul_107, parameter_25

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_183 = paddle._C_ops.add(add_182, add_171)
        del add_182

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_42 = paddle._C_ops.mean(add_183, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_42 = paddle._C_ops.subtract(add_183, mean_42)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_21 = paddle._C_ops.pow(subtract_42, float("2"))
        del subtract_42

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_43 = paddle._C_ops.mean(pow_21, full_int_array_2, True)
        del pow_21

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_43 = paddle._C_ops.subtract(add_183, mean_42)
        del add_183, mean_42

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_57 = paddle._C_ops.scale(mean_43, full_4, float("1e-07"), True)
        del mean_43

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_43 = paddle._C_ops.sqrt(scale_57)
        del scale_57

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_43 = paddle._C_ops.divide(subtract_43, sqrt_43)
        del sqrt_43, subtract_43

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_23 = paddle._C_ops.multiply(parameter_24, divide_43)
        del divide_43, parameter_24

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_184 = paddle._C_ops.add(multiply_23, parameter_23)
        del multiply_23, parameter_23

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_108 = paddle._C_ops.matmul(add_184, parameter_22, False, False)
        del parameter_22

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_185 = paddle._C_ops.add(matmul_108, parameter_21)
        del matmul_108, parameter_21

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_10 = paddle._C_ops.gelu(add_185, False)
        del add_185

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_109 = paddle._C_ops.matmul(gelu_10, parameter_20, False, False)
        del gelu_10, parameter_20

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_186 = paddle._C_ops.add(matmul_109, parameter_19)
        del matmul_109, parameter_19

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_187 = paddle._C_ops.add(add_186, add_184)
        del add_184, add_186

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_44 = paddle._C_ops.mean(add_187, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_44 = paddle._C_ops.subtract(add_187, mean_44)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_22 = paddle._C_ops.pow(subtract_44, float("2"))
        del subtract_44

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_45 = paddle._C_ops.mean(pow_22, full_int_array_2, True)
        del pow_22

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_45 = paddle._C_ops.subtract(add_187, mean_44)
        del add_187, mean_44

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_58 = paddle._C_ops.scale(mean_45, full_4, float("1e-07"), True)
        del mean_45

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_44 = paddle._C_ops.sqrt(scale_58)
        del scale_58

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_44 = paddle._C_ops.divide(subtract_45, sqrt_44)
        del sqrt_44, subtract_45

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_24 = paddle._C_ops.multiply(parameter_18, divide_44)
        del divide_44, parameter_18

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_188 = paddle._C_ops.add(multiply_24, parameter_17)
        del multiply_24, parameter_17

        # pd_op.matmul: (1x21x2304xf32) <- (1x21x768xf32, 768x2304xf32)
        matmul_110 = paddle._C_ops.matmul(add_188, parameter_14, False, False)
        del parameter_14

        # pd_op.reshape: (1x21x12x192xf32) <- (1x21x2304xf32, 4xi64)
        reshape_66 = paddle._C_ops.reshape(matmul_110, full_int_array_7)
        del full_int_array_7, matmul_110

        # pd_op.transpose: (1x12x21x192xf32) <- (1x21x12x192xf32)
        transpose_110 = paddle._C_ops.transpose(reshape_66, [0, 2, 1, 3])
        del reshape_66

        # pd_op.split_with_num: ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32]) <- (1x12x21x192xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(transpose_110, 3, full_5)
        del full_5, transpose_110

        # builtin.split: (1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32) <- ([1x12x21x64xf32, 1x12x21x64xf32, 1x12x21x64xf32])
        (
            split_33,
            split_34,
            split_35,
        ) = split_with_num_11
        del split_with_num_11

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_41 = paddle._C_ops.unsqueeze(parameter_16, full_int_array_8)
        del parameter_16

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_67 = paddle._C_ops.reshape(unsqueeze_41, full_int_array_9)
        del unsqueeze_41

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_111 = paddle._C_ops.transpose(reshape_67, [0, 2, 1, 3])
        del reshape_67

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_189 = paddle._C_ops.add(split_33, transpose_111)
        del split_33, transpose_111

        # pd_op.unsqueeze: (1x1x768xf32) <- (768xf32, 2xi64)
        unsqueeze_42 = paddle._C_ops.unsqueeze(parameter_15, full_int_array_8)
        del full_int_array_8, parameter_15

        # pd_op.reshape: (1x1x12x64xf32) <- (1x1x768xf32, 4xi64)
        reshape_68 = paddle._C_ops.reshape(unsqueeze_42, full_int_array_9)
        del full_int_array_9, unsqueeze_42

        # pd_op.transpose: (1x12x1x64xf32) <- (1x1x12x64xf32)
        transpose_112 = paddle._C_ops.transpose(reshape_68, [0, 2, 1, 3])
        del reshape_68

        # pd_op.add: (1x12x21x64xf32) <- (1x12x21x64xf32, 1x12x1x64xf32)
        add_190 = paddle._C_ops.add(split_35, transpose_112)
        del split_35, transpose_112

        # pd_op.full: (xi64) <- ()
        full_33 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__22 = paddle._C_ops.assign_value_(
            full_33,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_33

        # pd_op.cast: (xf32) <- (xi64)
        cast_38 = paddle._C_ops.cast(assign_value__22, paddle.float32)
        del assign_value__22

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_59 = paddle._C_ops.scale(cast_38, full_7, float("0"), True)
        del cast_38

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_45 = paddle._C_ops.sqrt(scale_59)
        del scale_59

        # pd_op.divide: (1x12x21x64xf32) <- (1x12x21x64xf32, xf32)
        divide_45 = paddle._C_ops.divide(add_189, sqrt_45)
        del add_189, sqrt_45

        # pd_op.transpose: (1x12x64x21xf32) <- (1x12x21x64xf32)
        transpose_113 = paddle._C_ops.transpose(split_34, [0, 1, 3, 2])

        # pd_op.matmul: (1x12x21x21xf32) <- (1x12x21x64xf32, 1x12x64x21xf32)
        matmul_111 = paddle._C_ops.matmul(divide_45, transpose_113, False, False)
        del transpose_113

        # pd_op.dropout: (1024x768xf32, 1024x768xui8) <- (1024x768xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                parameter_0, None, full_8, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del full_8, parameter_0

        # pd_op.slice: (42x768xf32) <- (1024x768xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            dropout_22, [0], full_int_array_10, full_int_array_11, [1], []
        )
        del dropout_22, full_int_array_10, full_int_array_11

        # pd_op.unsqueeze: (1x42x768xf32) <- (42x768xf32, 1xi64)
        unsqueeze_43 = paddle._C_ops.unsqueeze(slice_11, full_int_array_0)
        del full_int_array_0, slice_11

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_112 = paddle._C_ops.matmul(unsqueeze_43, parameter_13, False, False)
        del parameter_13

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_69 = paddle._C_ops.reshape(matmul_112, full_int_array_12)
        del matmul_112

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_114 = paddle._C_ops.transpose(reshape_69, [0, 2, 1, 3])
        del reshape_69

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_115 = paddle._C_ops.transpose(transpose_114, [0, 1, 3, 2])
        del transpose_114

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_113 = paddle._C_ops.matmul(divide_45, transpose_115, False, False)
        del divide_45, transpose_115

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_27 = paddle._C_ops.expand(matmul_113, full_int_array_14)
        del matmul_113

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_22 = paddle._C_ops.take_along_axis(expand_27, expand_2, 3)
        del expand_2, expand_27

        # pd_op.scale: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        scale_60 = paddle._C_ops.scale(take_along_axis_22, full_4, float("0"), True)
        del take_along_axis_22

        # pd_op.matmul: (1x42x768xf32) <- (1x42x768xf32, 768x768xf32)
        matmul_114 = paddle._C_ops.matmul(unsqueeze_43, parameter_12, False, False)
        del parameter_12, unsqueeze_43

        # pd_op.add: (1x42x768xf32) <- (1x42x768xf32, 768xf32)
        add_191 = paddle._C_ops.add(matmul_114, parameter_11)
        del matmul_114, parameter_11

        # pd_op.reshape: (1x42x12x64xf32) <- (1x42x768xf32, 4xi64)
        reshape_70 = paddle._C_ops.reshape(add_191, full_int_array_12)
        del add_191, full_int_array_12

        # pd_op.transpose: (1x12x42x64xf32) <- (1x42x12x64xf32)
        transpose_116 = paddle._C_ops.transpose(reshape_70, [0, 2, 1, 3])
        del reshape_70

        # pd_op.full: (xi64) <- ()
        full_34 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (xi64) <- (xi64)
        assign_value__23 = paddle._C_ops.assign_value_(
            full_34,
            [],
            paddle.int64,
            [float("64")],
            paddle.framework._current_expected_place(),
        )
        del full_34

        # pd_op.cast: (xf32) <- (xi64)
        cast_39 = paddle._C_ops.cast(assign_value__23, paddle.float32)
        del assign_value__23

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_61 = paddle._C_ops.scale(cast_39, full_7, float("0"), True)
        del cast_39, full_7

        # pd_op.sqrt: (xf32) <- (xf32)
        sqrt_46 = paddle._C_ops.sqrt(scale_61)
        del scale_61

        # pd_op.divide: (1x12x42x64xf32) <- (1x12x42x64xf32, xf32)
        divide_46 = paddle._C_ops.divide(transpose_116, sqrt_46)
        del sqrt_46, transpose_116

        # pd_op.transpose: (1x12x64x42xf32) <- (1x12x42x64xf32)
        transpose_117 = paddle._C_ops.transpose(divide_46, [0, 1, 3, 2])
        del divide_46

        # pd_op.matmul: (1x12x21x42xf32) <- (1x12x21x64xf32, 1x12x64x42xf32)
        matmul_115 = paddle._C_ops.matmul(split_34, transpose_117, False, False)
        del split_34, transpose_117

        # pd_op.expand: (1x12x21x42xf32) <- (1x12x21x42xf32, 4xi64)
        expand_28 = paddle._C_ops.expand(matmul_115, full_int_array_14)
        del full_int_array_14, matmul_115

        # pd_op.take_along_axis: (1x12x21x21xf32) <- (1x12x21x42xf32, 1x12x21x21xi64)
        take_along_axis_23 = paddle._C_ops.take_along_axis(expand_28, expand_5, 3)
        del expand_28, expand_5

        # pd_op.transpose: (1x12x21x21xf32) <- (1x12x21x21xf32)
        transpose_118 = paddle._C_ops.transpose(take_along_axis_23, [0, 1, 3, 2])
        del take_along_axis_23

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_192 = paddle._C_ops.add(scale_60, transpose_118)
        del scale_60, transpose_118

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_193 = paddle._C_ops.add(matmul_111, add_192)
        del add_192, matmul_111

        # pd_op.full_like: (1x12x21x21xf32) <- (1x12x21x21xf32, 1xf32)
        full_like_14 = paddle._C_ops.full_like(
            add_193, full_3, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_3

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_194 = paddle._C_ops.add(full_like_1, full_like_14)
        del full_like_1, full_like_14

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x1x21x21xf32)
        add_195 = paddle._C_ops.add(add_194, cast_5)
        del add_194, cast_5

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_196 = paddle._C_ops.add(full_12, add_195)
        del full_12

        # pd_op.add: (1x12x21x21xf32) <- (1x12x21x21xf32, 1x12x21x21xf32)
        add_197 = paddle._C_ops.add(add_193, add_195)
        del add_193

        # pd_op.add: (1x12x21x21xf32) <- (1x1x21x21xf32, 1x12x21x21xf32)
        add_198 = paddle._C_ops.add(cast_6, add_195)
        del add_195, cast_6

        # pd_op.cast: (1x12x21x21xb) <- (1x12x21x21xf32)
        cast_40 = paddle._C_ops.cast(add_198, paddle.bool)
        del add_198

        # pd_op.where: (1x12x21x21xf32) <- (1x12x21x21xb, 1x12x21x21xf32, 1x12x21x21xf32)
        where_11 = paddle._C_ops.where(cast_40, add_196, add_197)
        del add_196, add_197, cast_40

        # pd_op.softmax: (1x12x21x21xf32) <- (1x12x21x21xf32)
        softmax_11 = paddle._C_ops.softmax(where_11, -1)
        del where_11

        # pd_op.matmul: (1x12x21x64xf32) <- (1x12x21x21xf32, 1x12x21x64xf32)
        matmul_116 = paddle._C_ops.matmul(softmax_11, add_190, False, False)
        del add_190, softmax_11

        # pd_op.transpose: (1x21x12x64xf32) <- (1x12x21x64xf32)
        transpose_119 = paddle._C_ops.transpose(matmul_116, [0, 2, 1, 3])
        del matmul_116

        # pd_op.reshape: (1x21x768xf32) <- (1x21x12x64xf32, 3xi64)
        reshape_71 = paddle._C_ops.reshape(transpose_119, full_int_array_15)
        del full_int_array_15, transpose_119

        # pd_op.matmul: (1x21x768xf32) <- (1x21x768xf32, 768x768xf32)
        matmul_117 = paddle._C_ops.matmul(reshape_71, parameter_10, False, False)
        del parameter_10, reshape_71

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_199 = paddle._C_ops.add(matmul_117, parameter_9)
        del matmul_117, parameter_9

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_200 = paddle._C_ops.add(add_199, add_188)
        del add_199

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_46 = paddle._C_ops.mean(add_200, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_46 = paddle._C_ops.subtract(add_200, mean_46)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_23 = paddle._C_ops.pow(subtract_46, float("2"))
        del subtract_46

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_47 = paddle._C_ops.mean(pow_23, full_int_array_2, True)
        del pow_23

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_47 = paddle._C_ops.subtract(add_200, mean_46)
        del add_200, mean_46

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_62 = paddle._C_ops.scale(mean_47, full_4, float("1e-07"), True)
        del mean_47

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_47 = paddle._C_ops.sqrt(scale_62)
        del scale_62

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_47 = paddle._C_ops.divide(subtract_47, sqrt_47)
        del sqrt_47, subtract_47

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_25 = paddle._C_ops.multiply(parameter_8, divide_47)
        del divide_47, parameter_8

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_201 = paddle._C_ops.add(multiply_25, parameter_7)
        del multiply_25, parameter_7

        # pd_op.matmul: (1x21x3072xf32) <- (1x21x768xf32, 768x3072xf32)
        matmul_118 = paddle._C_ops.matmul(add_201, parameter_6, False, False)
        del parameter_6

        # pd_op.add: (1x21x3072xf32) <- (1x21x3072xf32, 3072xf32)
        add_202 = paddle._C_ops.add(matmul_118, parameter_5)
        del matmul_118, parameter_5

        # pd_op.gelu: (1x21x3072xf32) <- (1x21x3072xf32)
        gelu_11 = paddle._C_ops.gelu(add_202, False)
        del add_202

        # pd_op.matmul: (1x21x768xf32) <- (1x21x3072xf32, 3072x768xf32)
        matmul_119 = paddle._C_ops.matmul(gelu_11, parameter_4, False, False)
        del gelu_11, parameter_4

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_203 = paddle._C_ops.add(matmul_119, parameter_3)
        del matmul_119, parameter_3

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 1x21x768xf32)
        add_204 = paddle._C_ops.add(add_203, add_201)
        del add_201, add_203

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_48 = paddle._C_ops.mean(add_204, full_int_array_2, True)

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_48 = paddle._C_ops.subtract(add_204, mean_48)

        # pd_op.pow: (1x21x768xf32) <- (1x21x768xf32)
        pow_24 = paddle._C_ops.pow(subtract_48, float("2"))
        del subtract_48

        # pd_op.mean: (1x21x1xf32) <- (1x21x768xf32, 1xi64)
        mean_49 = paddle._C_ops.mean(pow_24, full_int_array_2, True)
        del full_int_array_2, pow_24

        # pd_op.subtract: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        subtract_49 = paddle._C_ops.subtract(add_204, mean_48)
        del add_204, mean_48

        # pd_op.scale: (1x21x1xf32) <- (1x21x1xf32, 1xf32)
        scale_63 = paddle._C_ops.scale(mean_49, full_4, float("1e-07"), True)
        del full_4, mean_49

        # pd_op.sqrt: (1x21x1xf32) <- (1x21x1xf32)
        sqrt_48 = paddle._C_ops.sqrt(scale_63)
        del scale_63

        # pd_op.divide: (1x21x768xf32) <- (1x21x768xf32, 1x21x1xf32)
        divide_48 = paddle._C_ops.divide(subtract_49, sqrt_48)
        del sqrt_48, subtract_49

        # pd_op.multiply: (1x21x768xf32) <- (768xf32, 1x21x768xf32)
        multiply_26 = paddle._C_ops.multiply(parameter_2, divide_48)
        del divide_48, parameter_2

        # pd_op.add: (1x21x768xf32) <- (1x21x768xf32, 768xf32)
        add_0 = paddle._C_ops.add(multiply_26, parameter_1)
        del (
            add_103,
            add_120,
            add_137,
            add_154,
            add_171,
            add_18,
            add_188,
            add_35,
            add_52,
            add_69,
            add_86,
            multiply_1,
            multiply_26,
            parameter_1,
        )

        return add_0
