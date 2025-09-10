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
        parameter_196,
        parameter_197,
        parameter_198,
        parameter_199,
        parameter_200,
        parameter_201,
        parameter_202,
        parameter_203,
        parameter_204,
        parameter_205,
        parameter_206,
        parameter_207,
        parameter_208,
        parameter_209,
        parameter_210,
        parameter_211,
        parameter_212,
        parameter_213,
        parameter_214,
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
    ):
        # pd_op.flatten: (2x512x196xf32) <- (2x512x14x14xf32)
        flatten_0 = paddle._C_ops.flatten(data_10, 2, 3)
        del data_10

        # pd_op.transpose: (2x196x512xf32) <- (2x512x196xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.full: (1xf64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("14"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (14xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype="float32")
        del full_1

        # builtin.combine: ([14xf32, 14xf32]) <- (14xf32, 14xf32)
        combine_0 = [arange_0, arange_0]
        del arange_0

        # pd_op.meshgrid: ([14x14xf32, 14x14xf32]) <- ([14xf32, 14xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (14x14xf32, 14x14xf32) <- ([14x14xf32, 14x14xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xf32) <- (1xf64, 1xf64, 1xf64)
        arange_1 = paddle.arange(full_0, full_3, full_2, dtype="float32")
        del full_0, full_2, full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(arange_1, full_4, float("0"), True)
        del arange_1, full_4

        # pd_op.full: (128xf32) <- ()
        full_5 = paddle._C_ops.full(
            [128],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_5, scale_0)
        del full_5, scale_0

        # pd_op.full: (128xf32) <- ()
        full_6 = paddle._C_ops.full(
            [128],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.divide: (128xf32) <- (128xf32, 128xf32)
        divide_0 = paddle._C_ops.divide(full_6, elementwise_pow_0)
        del elementwise_pow_0, full_6

        # pd_op.flatten: (196xf32) <- (14x14xf32)
        flatten_1 = paddle._C_ops.flatten(split_0, 0, 1)
        del split_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (196x1xf32) <- (196xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(flatten_1, full_int_array_0)
        del flatten_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

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

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_1

        # pd_op.unsqueeze: (1x128xf32) <- (128xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(divide_0, full_int_array_1)
        del divide_0

        # pd_op.matmul: (196x128xf32) <- (196x1xf32, 1x128xf32)
        matmul_0 = paddle._C_ops.matmul(unsqueeze_0, unsqueeze_1, False, False)
        del unsqueeze_0

        # pd_op.flatten: (196xf32) <- (14x14xf32)
        flatten_2 = paddle._C_ops.flatten(split_1, 0, 1)
        del split_1

        # pd_op.unsqueeze: (196x1xf32) <- (196xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(flatten_2, full_int_array_0)
        del flatten_2, full_int_array_0

        # pd_op.matmul: (196x128xf32) <- (196x1xf32, 1x128xf32)
        matmul_1 = paddle._C_ops.matmul(unsqueeze_2, unsqueeze_1, False, False)
        del unsqueeze_1, unsqueeze_2

        # pd_op.sin: (196x128xf32) <- (196x128xf32)
        sin_0 = paddle._C_ops.sin(matmul_0)

        # pd_op.cos: (196x128xf32) <- (196x128xf32)
        cos_0 = paddle._C_ops.cos(matmul_0)
        del matmul_0

        # pd_op.sin: (196x128xf32) <- (196x128xf32)
        sin_1 = paddle._C_ops.sin(matmul_1)

        # pd_op.cos: (196x128xf32) <- (196x128xf32)
        cos_1 = paddle._C_ops.cos(matmul_1)
        del matmul_1

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_7

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_7

        # builtin.combine: ([196x128xf32, 196x128xf32, 196x128xf32, 196x128xf32]) <- (196x128xf32, 196x128xf32, 196x128xf32, 196x128xf32)
        combine_1 = [sin_0, cos_0, sin_1, cos_1]
        del cos_0, cos_1, sin_0, sin_1

        # pd_op.concat: (196x512xf32) <- ([196x128xf32, 196x128xf32, 196x128xf32, 196x128xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_7)
        del combine_1

        # pd_op.unsqueeze: (1x196x512xf32) <- (196x512xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(concat_0, full_int_array_1)
        del concat_0

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 1x196x512xf32)
        add_0 = paddle._C_ops.add(transpose_0, unsqueeze_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [512]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_18 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_19 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_20 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_21 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_22 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_23 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_24 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_25 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_26 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_27 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_28 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_29 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_30 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_31 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_32 = full_int_array_2

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_2 = paddle._C_ops.matmul(add_0, slice_0, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_1 = paddle._C_ops.add(matmul_2, slice_1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [0, 0, 4, 128]

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(add_1, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_0, [0, 2, 1, 3])
        del reshape_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1024]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_33 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_34 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_35 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_36 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_37 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_38 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_39 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_40 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_41 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_42 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_43 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_44 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_45 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_46 = full_int_array_4

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_47 = full_int_array_4

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_0, [1], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_1, [0], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_3 = paddle._C_ops.matmul(add_0, slice_2, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_2 = paddle._C_ops.add(matmul_3, slice_3)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_1, [0, 2, 1, 3])
        del reshape_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_48 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_49 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_50 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_51 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_52 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_53 = full_int_array_5

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_54 = full_int_array_5

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_0, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del data_0

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_1, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del data_1

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_4 = paddle._C_ops.matmul(transpose_0, slice_4, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_3 = paddle._C_ops.add(matmul_4, slice_5)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (2x4x196x196xf32) <- (2x4x196x128xf32, 2x4x196x128xf32)
        matmul_5 = paddle._C_ops.matmul(transpose_1, transpose_2, False, True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0.0883883"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_55 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_56 = full_8

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_57 = full_8

        # pd_op.scale: (2x4x196x196xf32) <- (2x4x196x196xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(matmul_5, full_8, float("0"), True)
        del matmul_5

        # pd_op.softmax: (2x4x196x196xf32) <- (2x4x196x196xf32)
        softmax_0 = paddle._C_ops.softmax(scale_1, -1)
        del scale_1

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_58 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_59 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_60 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_61 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_62 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_63 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_64 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_65 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_66 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_67 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_68 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_69 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_70 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_71 = full_9

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_72 = full_9

        # pd_op.dropout: (2x4x196x196xf32, 2x4x196x196xui8) <- (2x4x196x196xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (2x4x196x128xf32) <- (2x4x196x196xf32, 2x4x196x128xf32)
        matmul_6 = paddle._C_ops.matmul(dropout_0, transpose_3, False, False)

        # pd_op.transpose: (2x196x4x128xf32) <- (2x4x196x128xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_6, [0, 2, 1, 3])
        del matmul_6

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [0, 0, 512]

        # pd_op.reshape: (2x196x512xf32) <- (2x196x4x128xf32, 3xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_4, full_int_array_6)

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_7 = paddle._C_ops.matmul(reshape_3, parameter_214, False, False)
        del parameter_214

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_4 = paddle._C_ops.add(matmul_7, parameter_213)
        del parameter_213

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_4, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_4

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_5 = paddle._C_ops.add(transpose_0, dropout_2)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_5, parameter_212, parameter_211, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_211, parameter_212

        # pd_op.matmul: (2x196x2048xf32) <- (2x196x512xf32, 512x2048xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_0, parameter_210, False, False)
        del parameter_210

        # pd_op.add: (2x196x2048xf32) <- (2x196x2048xf32, 2048xf32)
        add_6 = paddle._C_ops.add(matmul_8, parameter_209)
        del parameter_209

        # pd_op.gelu: (2x196x2048xf32) <- (2x196x2048xf32)
        gelu_0 = paddle._C_ops.gelu(add_6, False)

        # pd_op.dropout: (2x196x2048xf32, 2x196x2048xui8) <- (2x196x2048xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_0, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_0

        # pd_op.matmul: (2x196x512xf32) <- (2x196x2048xf32, 2048x512xf32)
        matmul_9 = paddle._C_ops.matmul(dropout_4, parameter_208, False, False)
        del parameter_208

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_7 = paddle._C_ops.add(matmul_9, parameter_207)
        del parameter_207

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_7, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_7

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_8 = paddle._C_ops.add(layer_norm_0, dropout_6)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_8, parameter_206, parameter_205, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_205, parameter_206

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 1x196x512xf32)
        add_9 = paddle._C_ops.add(layer_norm_3, unsqueeze_3)

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_3, [0], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_10 = paddle._C_ops.matmul(add_9, slice_6, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_10 = paddle._C_ops.add(matmul_10, slice_7)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_10, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_3, [0], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_11 = paddle._C_ops.matmul(add_9, slice_8, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_11 = paddle._C_ops.add(matmul_11, slice_9)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(add_11, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_5, [0, 2, 1, 3])
        del reshape_5

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_2, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del data_2

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_3, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del data_3

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_3, slice_10, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_12 = paddle._C_ops.add(matmul_12, slice_11)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_12, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (2x4x196x196xf32) <- (2x4x196x128xf32, 2x4x196x128xf32)
        matmul_13 = paddle._C_ops.matmul(transpose_5, transpose_6, False, True)

        # pd_op.scale: (2x4x196x196xf32) <- (2x4x196x196xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(matmul_13, full_8, float("0"), True)
        del matmul_13

        # pd_op.softmax: (2x4x196x196xf32) <- (2x4x196x196xf32)
        softmax_1 = paddle._C_ops.softmax(scale_2, -1)
        del scale_2

        # pd_op.dropout: (2x4x196x196xf32, 2x4x196x196xui8) <- (2x4x196x196xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (2x4x196x128xf32) <- (2x4x196x196xf32, 2x4x196x128xf32)
        matmul_14 = paddle._C_ops.matmul(dropout_8, transpose_7, False, False)

        # pd_op.transpose: (2x196x4x128xf32) <- (2x4x196x128xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])
        del matmul_14

        # pd_op.reshape: (2x196x512xf32) <- (2x196x4x128xf32, 3xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_8, full_int_array_6)

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_15 = paddle._C_ops.matmul(reshape_7, parameter_204, False, False)
        del parameter_204

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_13 = paddle._C_ops.add(matmul_15, parameter_203)
        del parameter_203

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_13, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_13

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_14 = paddle._C_ops.add(layer_norm_3, dropout_10)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_14, parameter_202, parameter_201, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_201, parameter_202

        # pd_op.matmul: (2x196x2048xf32) <- (2x196x512xf32, 512x2048xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_6, parameter_200, False, False)
        del parameter_200

        # pd_op.add: (2x196x2048xf32) <- (2x196x2048xf32, 2048xf32)
        add_15 = paddle._C_ops.add(matmul_16, parameter_199)
        del parameter_199

        # pd_op.gelu: (2x196x2048xf32) <- (2x196x2048xf32)
        gelu_1 = paddle._C_ops.gelu(add_15, False)

        # pd_op.dropout: (2x196x2048xf32, 2x196x2048xui8) <- (2x196x2048xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_1, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_1

        # pd_op.matmul: (2x196x512xf32) <- (2x196x2048xf32, 2048x512xf32)
        matmul_17 = paddle._C_ops.matmul(dropout_12, parameter_198, False, False)
        del parameter_198

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_16 = paddle._C_ops.add(matmul_17, parameter_197)
        del parameter_197

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_16, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_16

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_17 = paddle._C_ops.add(layer_norm_6, dropout_14)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_196, parameter_195, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_195, parameter_196

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 1x196x512xf32)
        add_18 = paddle._C_ops.add(layer_norm_9, unsqueeze_3)

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_4, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            data_5, [0], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_18 = paddle._C_ops.matmul(add_18, slice_12, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_19 = paddle._C_ops.add(matmul_18, slice_13)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_19, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_9 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            data_4, [1], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            data_5, [0], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_19 = paddle._C_ops.matmul(add_18, slice_14, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_20 = paddle._C_ops.add(matmul_19, slice_15)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(add_20, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_9, [0, 2, 1, 3])
        del reshape_9

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            data_4, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del data_4

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            data_5, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del data_5

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_20 = paddle._C_ops.matmul(layer_norm_9, slice_16, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_21 = paddle._C_ops.add(matmul_20, slice_17)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_21, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (2x4x196x196xf32) <- (2x4x196x128xf32, 2x4x196x128xf32)
        matmul_21 = paddle._C_ops.matmul(transpose_9, transpose_10, False, True)

        # pd_op.scale: (2x4x196x196xf32) <- (2x4x196x196xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(matmul_21, full_8, float("0"), True)
        del matmul_21

        # pd_op.softmax: (2x4x196x196xf32) <- (2x4x196x196xf32)
        softmax_2 = paddle._C_ops.softmax(scale_3, -1)
        del scale_3

        # pd_op.dropout: (2x4x196x196xf32, 2x4x196x196xui8) <- (2x4x196x196xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (2x4x196x128xf32) <- (2x4x196x196xf32, 2x4x196x128xf32)
        matmul_22 = paddle._C_ops.matmul(dropout_16, transpose_11, False, False)

        # pd_op.transpose: (2x196x4x128xf32) <- (2x4x196x128xf32)
        transpose_12 = paddle._C_ops.transpose(matmul_22, [0, 2, 1, 3])
        del matmul_22

        # pd_op.reshape: (2x196x512xf32) <- (2x196x4x128xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_12, full_int_array_6)

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_23 = paddle._C_ops.matmul(reshape_11, parameter_194, False, False)
        del parameter_194

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_22 = paddle._C_ops.add(matmul_23, parameter_193)
        del parameter_193

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_22, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_22

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_23 = paddle._C_ops.add(layer_norm_9, dropout_18)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_23, parameter_192, parameter_191, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_191, parameter_192

        # pd_op.matmul: (2x196x2048xf32) <- (2x196x512xf32, 512x2048xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_12, parameter_190, False, False)
        del parameter_190

        # pd_op.add: (2x196x2048xf32) <- (2x196x2048xf32, 2048xf32)
        add_24 = paddle._C_ops.add(matmul_24, parameter_189)
        del parameter_189

        # pd_op.gelu: (2x196x2048xf32) <- (2x196x2048xf32)
        gelu_2 = paddle._C_ops.gelu(add_24, False)

        # pd_op.dropout: (2x196x2048xf32, 2x196x2048xui8) <- (2x196x2048xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_2, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_2

        # pd_op.matmul: (2x196x512xf32) <- (2x196x2048xf32, 2048x512xf32)
        matmul_25 = paddle._C_ops.matmul(dropout_20, parameter_188, False, False)
        del parameter_188

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_25 = paddle._C_ops.add(matmul_25, parameter_187)
        del parameter_187

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_25, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_25

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_26 = paddle._C_ops.add(layer_norm_12, dropout_22)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_186, parameter_185, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_185, parameter_186

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 1x196x512xf32)
        add_27 = paddle._C_ops.add(layer_norm_15, unsqueeze_3)

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            data_6, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            data_7, [0], full_int_array_1, full_int_array_2, [1], []
        )
        del full_int_array_1

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_26 = paddle._C_ops.matmul(add_27, slice_18, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_28 = paddle._C_ops.add(matmul_26, slice_19)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_28, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_13 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            data_6, [1], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            data_7, [0], full_int_array_2, full_int_array_4, [1], []
        )

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_27 = paddle._C_ops.matmul(add_27, slice_20, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_29 = paddle._C_ops.add(matmul_27, slice_21)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(add_29, full_int_array_3)

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_13, [0, 2, 1, 3])
        del reshape_13

        # pd_op.slice: (512x512xf32) <- (512x1536xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            data_6, [1], full_int_array_4, full_int_array_5, [1], []
        )
        del data_6

        # pd_op.slice: (512xf32) <- (1536xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            data_7, [0], full_int_array_4, full_int_array_5, [1], []
        )
        del data_7

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_15, slice_22, False, False)

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_30 = paddle._C_ops.add(matmul_28, slice_23)

        # pd_op.reshape: (2x196x4x128xf32) <- (2x196x512xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_30, full_int_array_3)
        del full_int_array_3

        # pd_op.transpose: (2x4x196x128xf32) <- (2x196x4x128xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (2x4x196x196xf32) <- (2x4x196x128xf32, 2x4x196x128xf32)
        matmul_29 = paddle._C_ops.matmul(transpose_13, transpose_14, False, True)

        # pd_op.scale: (2x4x196x196xf32) <- (2x4x196x196xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(matmul_29, full_8, float("0"), True)
        del matmul_29

        # pd_op.softmax: (2x4x196x196xf32) <- (2x4x196x196xf32)
        softmax_3 = paddle._C_ops.softmax(scale_4, -1)
        del scale_4

        # pd_op.dropout: (2x4x196x196xf32, 2x4x196x196xui8) <- (2x4x196x196xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )

        # pd_op.matmul: (2x4x196x128xf32) <- (2x4x196x196xf32, 2x4x196x128xf32)
        matmul_30 = paddle._C_ops.matmul(dropout_24, transpose_15, False, False)

        # pd_op.transpose: (2x196x4x128xf32) <- (2x4x196x128xf32)
        transpose_16 = paddle._C_ops.transpose(matmul_30, [0, 2, 1, 3])
        del matmul_30

        # pd_op.reshape: (2x196x512xf32) <- (2x196x4x128xf32, 3xi64)
        reshape_15 = paddle._C_ops.reshape(transpose_16, full_int_array_6)
        del full_int_array_6

        # pd_op.matmul: (2x196x512xf32) <- (2x196x512xf32, 512x512xf32)
        matmul_31 = paddle._C_ops.matmul(reshape_15, parameter_184, False, False)
        del parameter_184

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_31 = paddle._C_ops.add(matmul_31, parameter_183)
        del parameter_183

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_31, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_31

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_32 = paddle._C_ops.add(layer_norm_15, dropout_26)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_182, parameter_181, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_181, parameter_182

        # pd_op.matmul: (2x196x2048xf32) <- (2x196x512xf32, 512x2048xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_18, parameter_180, False, False)
        del parameter_180

        # pd_op.add: (2x196x2048xf32) <- (2x196x2048xf32, 2048xf32)
        add_33 = paddle._C_ops.add(matmul_32, parameter_179)
        del parameter_179

        # pd_op.gelu: (2x196x2048xf32) <- (2x196x2048xf32)
        gelu_3 = paddle._C_ops.gelu(add_33, False)

        # pd_op.dropout: (2x196x2048xf32, 2x196x2048xui8) <- (2x196x2048xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                gelu_3, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del gelu_3

        # pd_op.matmul: (2x196x512xf32) <- (2x196x2048xf32, 2048x512xf32)
        matmul_33 = paddle._C_ops.matmul(dropout_28, parameter_178, False, False)
        del parameter_178

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 512xf32)
        add_34 = paddle._C_ops.add(matmul_33, parameter_177)
        del parameter_177

        # pd_op.dropout: (2x196x512xf32, 2x196x512xui8) <- (2x196x512xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_34, None, full_9, False, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_34

        # pd_op.add: (2x196x512xf32) <- (2x196x512xf32, 2x196x512xf32)
        add_35 = paddle._C_ops.add(layer_norm_18, dropout_30)

        # pd_op.layer_norm: (2x196x512xf32, 2x196xf32, 2x196xf32) <- (2x196x512xf32, 512xf32, 512xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_176, parameter_175, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del parameter_175, parameter_176

        # pd_op.transpose: (2x512x196xf32) <- (2x196x512xf32)
        transpose_17 = paddle._C_ops.transpose(layer_norm_21, [0, 2, 1])
        del layer_norm_21

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [2, 512, 14, 14]

        # pd_op.reshape: (2x512x14x14xf32) <- (2x512x196xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(transpose_17, full_int_array_7)
        del full_int_array_7

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x512x14x14xf32, 192x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            reshape_16, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_174

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_173,
                parameter_172,
                parameter_171,
                parameter_170,
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
        del parameter_170, parameter_171, parameter_172, parameter_173

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x512x14x14xf32, 192x512x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_16, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_169

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_168,
                parameter_167,
                parameter_166,
                parameter_165,
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
        del parameter_165, parameter_166, parameter_167, parameter_168

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            swish_2, parameter_164, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_164

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_163,
                parameter_162,
                parameter_161,
                parameter_160,
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
        del parameter_160, parameter_161, parameter_162, parameter_163

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            swish_3, parameter_159, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_159

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_3,
                parameter_158,
                parameter_157,
                parameter_156,
                parameter_155,
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
        del parameter_155, parameter_156, parameter_157, parameter_158

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_3, parameter_154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_154

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_153,
                parameter_152,
                parameter_151,
                parameter_150,
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
        del parameter_150, parameter_151, parameter_152, parameter_153

        # pd_op.add: (2x192x14x14xf32) <- (2x192x14x14xf32, 2x192x14x14xf32)
        add_36 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_4 = paddle._C_ops.swish(add_36)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [5, 5]

        # pd_op.pool2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            swish_4,
            full_int_array_8,
            [1, 1],
            [2, 2],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [9, 9]

        # pd_op.pool2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            swish_4,
            full_int_array_9,
            [1, 1],
            [4, 4],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [13, 13]

        # pd_op.pool2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            swish_4,
            full_int_array_10,
            [1, 1],
            [6, 6],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )

        # builtin.combine: ([2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32]) <- (2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32)
        combine_2 = [swish_4, pool2d_0, pool2d_1, pool2d_2]

        # pd_op.concat: (2x768x14x14xf32) <- ([2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32, 2x192x14x14xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_7)
        del combine_2

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x768x14x14xf32, 192x768x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_1, parameter_149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_149

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_148,
                parameter_147,
                parameter_146,
                parameter_145,
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
        del parameter_145, parameter_146, parameter_147, parameter_148

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__30)

        # builtin.combine: ([2x192x14x14xf32, 2x192x14x14xf32]) <- (2x192x14x14xf32, 2x192x14x14xf32)
        combine_3 = [swish_1, swish_5]

        # pd_op.concat: (2x384x14x14xf32) <- ([2x192x14x14xf32, 2x192x14x14xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_7)
        del combine_3

        # pd_op.conv2d: (2x384x14x14xf32) <- (2x384x14x14xf32, 384x384x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_2, parameter_144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_144

        # pd_op.batch_norm_: (2x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
                parameter_143,
                parameter_142,
                parameter_141,
                parameter_140,
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
        del parameter_140, parameter_141, parameter_142, parameter_143

        # pd_op.swish: (2x384x14x14xf32) <- (2x384x14x14xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x384x14x14xf32, 192x384x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            swish_6, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_139

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_7,
                parameter_138,
                parameter_137,
                parameter_136,
                parameter_135,
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
        del parameter_135, parameter_136, parameter_137, parameter_138

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__42)

        # pd_op.nearest_interp: (2x192x28x28xf32) <- (2x192x14x14xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            swish_7,
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

        # builtin.combine: ([2x192x28x28xf32, 2x256x-1x-1xf32]) <- (2x192x28x28xf32, 2x256x-1x-1xf32)
        combine_4 = [nearest_interp_0, data_9]
        del data_9

        # pd_op.concat: (2x448x28x28xf32) <- ([2x192x28x28xf32, 2x256x-1x-1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_4, full_7)
        del combine_4

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x448x28x28xf32, 96x448x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_3, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_134

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
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
        del parameter_130, parameter_131, parameter_132, parameter_133

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_8 = paddle._C_ops.swish(batch_norm__48)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x448x28x28xf32, 96x448x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_3, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_129

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
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
        del parameter_125, parameter_126, parameter_127, parameter_128

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            swish_9, parameter_124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_124

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
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
        del parameter_120, parameter_121, parameter_122, parameter_123

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            swish_10, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_119

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
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
        del parameter_115, parameter_116, parameter_117, parameter_118

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            swish_10, parameter_114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_114

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
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
        del parameter_110, parameter_111, parameter_112, parameter_113

        # pd_op.add: (2x96x28x28xf32) <- (2x96x28x28xf32, 2x96x28x28xf32)
        add_37 = paddle._C_ops.add(batch_norm__66, batch_norm__72)

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_11 = paddle._C_ops.swish(add_37)

        # builtin.combine: ([2x96x28x28xf32, 2x96x28x28xf32]) <- (2x96x28x28xf32, 2x96x28x28xf32)
        combine_5 = [swish_8, swish_11]

        # pd_op.concat: (2x192x28x28xf32) <- ([2x96x28x28xf32, 2x96x28x28xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_5, full_7)
        del combine_5

        # pd_op.conv2d: (2x192x28x28xf32) <- (2x192x28x28xf32, 192x192x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_4, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_109

        # pd_op.batch_norm_: (2x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_108,
                parameter_107,
                parameter_106,
                parameter_105,
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
        del parameter_105, parameter_106, parameter_107, parameter_108

        # pd_op.swish: (2x192x28x28xf32) <- (2x192x28x28xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x192x28x28xf32, 96x192x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            swish_12, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_104

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
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
        del parameter_100, parameter_101, parameter_102, parameter_103

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__84)

        # pd_op.nearest_interp: (2x96x56x56xf32) <- (2x96x28x28xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            swish_13,
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

        # builtin.combine: ([2x96x56x56xf32, 2x128x-1x-1xf32]) <- (2x96x56x56xf32, 2x128x-1x-1xf32)
        combine_6 = [nearest_interp_1, data_8]
        del data_8

        # pd_op.concat: (2x224x56x56xf32) <- ([2x96x56x56xf32, 2x128x-1x-1xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_6, full_7)
        del combine_6

        # pd_op.conv2d: (2x48x56x56xf32) <- (2x224x56x56xf32, 48x224x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_5, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_99

        # pd_op.batch_norm_: (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
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
        del parameter_95, parameter_96, parameter_97, parameter_98

        # pd_op.swish: (2x48x56x56xf32) <- (2x48x56x56xf32)
        swish_14 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.conv2d: (2x48x56x56xf32) <- (2x224x56x56xf32, 48x224x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_5, parameter_94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_94

        # pd_op.batch_norm_: (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
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
        del parameter_90, parameter_91, parameter_92, parameter_93

        # pd_op.swish: (2x48x56x56xf32) <- (2x48x56x56xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.conv2d: (2x48x56x56xf32) <- (2x48x56x56xf32, 48x48x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            swish_15, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89

        # pd_op.batch_norm_: (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
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
        del parameter_85, parameter_86, parameter_87, parameter_88

        # pd_op.swish: (2x48x56x56xf32) <- (2x48x56x56xf32)
        swish_16 = paddle._C_ops.swish(batch_norm__102)

        # pd_op.conv2d: (2x48x56x56xf32) <- (2x48x56x56xf32, 48x48x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            swish_16, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84

        # pd_op.batch_norm_: (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
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
        del parameter_80, parameter_81, parameter_82, parameter_83

        # pd_op.conv2d: (2x48x56x56xf32) <- (2x48x56x56xf32, 48x48x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            swish_16, parameter_79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_79

        # pd_op.batch_norm_: (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x56x56xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
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
        del parameter_75, parameter_76, parameter_77, parameter_78

        # pd_op.add: (2x48x56x56xf32) <- (2x48x56x56xf32, 2x48x56x56xf32)
        add_38 = paddle._C_ops.add(batch_norm__108, batch_norm__114)

        # pd_op.swish: (2x48x56x56xf32) <- (2x48x56x56xf32)
        swish_17 = paddle._C_ops.swish(add_38)

        # builtin.combine: ([2x48x56x56xf32, 2x48x56x56xf32]) <- (2x48x56x56xf32, 2x48x56x56xf32)
        combine_7 = [swish_14, swish_17]

        # pd_op.concat: (2x96x56x56xf32) <- ([2x48x56x56xf32, 2x48x56x56xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_7, full_7)
        del combine_7

        # pd_op.conv2d: (2x96x56x56xf32) <- (2x96x56x56xf32, 96x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            concat_6, parameter_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_74

        # pd_op.batch_norm_: (2x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
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
        del parameter_70, parameter_71, parameter_72, parameter_73

        # pd_op.swish: (2x96x56x56xf32) <- (2x96x56x56xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__120)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x56x56xf32, 96x96x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            swish_18, parameter_69, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
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
        del parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__126)

        # builtin.combine: ([2x96x28x28xf32, 2x192x28x28xf32]) <- (2x96x28x28xf32, 2x192x28x28xf32)
        combine_8 = [swish_19, swish_12]

        # pd_op.concat: (2x288x28x28xf32) <- ([2x96x28x28xf32, 2x192x28x28xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_8, full_7)
        del combine_8

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x288x28x28xf32, 96x288x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_7, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
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
        del parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_20 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x288x28x28xf32, 96x288x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            concat_7, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_59

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
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
        del parameter_55, parameter_56, parameter_57, parameter_58

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_21 = paddle._C_ops.swish(batch_norm__138)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            swish_21, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
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
        del parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__144)

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            swish_22, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
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
        del parameter_45, parameter_46, parameter_47, parameter_48

        # pd_op.conv2d: (2x96x28x28xf32) <- (2x96x28x28xf32, 96x96x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            swish_22, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_44

        # pd_op.batch_norm_: (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x28x28xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
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
        del parameter_40, parameter_41, parameter_42, parameter_43

        # pd_op.add: (2x96x28x28xf32) <- (2x96x28x28xf32, 2x96x28x28xf32)
        add_39 = paddle._C_ops.add(batch_norm__150, batch_norm__156)

        # pd_op.swish: (2x96x28x28xf32) <- (2x96x28x28xf32)
        swish_23 = paddle._C_ops.swish(add_39)

        # builtin.combine: ([2x96x28x28xf32, 2x96x28x28xf32]) <- (2x96x28x28xf32, 2x96x28x28xf32)
        combine_9 = [swish_20, swish_23]

        # pd_op.concat: (2x192x28x28xf32) <- ([2x96x28x28xf32, 2x96x28x28xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_9, full_7)
        del combine_9

        # pd_op.conv2d: (2x192x28x28xf32) <- (2x192x28x28xf32, 192x192x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            concat_8, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39

        # pd_op.batch_norm_: (2x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x28x28xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
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
        del parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.swish: (2x192x28x28xf32) <- (2x192x28x28xf32)
        swish_24 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x28x28xf32, 192x192x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            swish_24, parameter_34, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
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
        del parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__168)

        # builtin.combine: ([2x192x14x14xf32, 2x384x14x14xf32]) <- (2x192x14x14xf32, 2x384x14x14xf32)
        combine_10 = [swish_25, swish_6]

        # pd_op.concat: (2x576x14x14xf32) <- ([2x192x14x14xf32, 2x384x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_10, full_7)
        del combine_10

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x576x14x14xf32, 192x576x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            concat_9, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
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
        del parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_26 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x576x14x14xf32, 192x576x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            concat_9, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
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
        del parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            swish_27, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
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
        del parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            swish_28, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_14

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
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
        del parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.conv2d: (2x192x14x14xf32) <- (2x192x14x14xf32, 192x192x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            swish_28, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.batch_norm_: (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x14x14xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
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
        del parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.add: (2x192x14x14xf32) <- (2x192x14x14xf32, 2x192x14x14xf32)
        add_40 = paddle._C_ops.add(batch_norm__192, batch_norm__198)

        # pd_op.swish: (2x192x14x14xf32) <- (2x192x14x14xf32)
        swish_29 = paddle._C_ops.swish(add_40)

        # builtin.combine: ([2x192x14x14xf32, 2x192x14x14xf32]) <- (2x192x14x14xf32, 2x192x14x14xf32)
        combine_11 = [swish_26, swish_29]

        # pd_op.concat: (2x384x14x14xf32) <- ([2x192x14x14xf32, 2x192x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_11, full_7)
        del combine_11, full_7

        # pd_op.conv2d: (2x384x14x14xf32) <- (2x384x14x14xf32, 384x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

        # pd_op.batch_norm_: (2x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x14x14xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
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
        del parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.swish: (2x384x14x14xf32) <- (2x384x14x14xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__204)
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_12,
            add_14,
            add_15,
            add_17,
            add_18,
            add_19,
            add_2,
            add_20,
            add_21,
            add_23,
            add_24,
            add_26,
            add_27,
            add_28,
            add_29,
            add_3,
            add_30,
            add_32,
            add_33,
            add_35,
            add_36,
            add_37,
            add_38,
            add_39,
            add_40,
            add_5,
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
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            batch_norm__109,
            batch_norm__11,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__12,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__13,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__138,
            batch_norm__139,
            batch_norm__14,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__15,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__16,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            batch_norm__168,
            batch_norm__169,
            batch_norm__17,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            batch_norm__18,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__19,
            batch_norm__190,
            batch_norm__191,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
            batch_norm__2,
            batch_norm__20,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__3,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__5,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__6,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__9,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            concat_1,
            concat_10,
            concat_2,
            concat_3,
            concat_4,
            concat_5,
            concat_6,
            concat_7,
            concat_8,
            concat_9,
            conv2d_0,
            conv2d_1,
            conv2d_10,
            conv2d_11,
            conv2d_12,
            conv2d_13,
            conv2d_14,
            conv2d_15,
            conv2d_16,
            conv2d_17,
            conv2d_18,
            conv2d_19,
            conv2d_2,
            conv2d_20,
            conv2d_21,
            conv2d_22,
            conv2d_23,
            conv2d_24,
            conv2d_25,
            conv2d_26,
            conv2d_27,
            conv2d_28,
            conv2d_29,
            conv2d_3,
            conv2d_30,
            conv2d_31,
            conv2d_32,
            conv2d_33,
            conv2d_34,
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            dropout_0,
            dropout_1,
            dropout_10,
            dropout_11,
            dropout_12,
            dropout_13,
            dropout_14,
            dropout_15,
            dropout_16,
            dropout_17,
            dropout_18,
            dropout_19,
            dropout_2,
            dropout_20,
            dropout_21,
            dropout_22,
            dropout_23,
            dropout_24,
            dropout_25,
            dropout_26,
            dropout_27,
            dropout_28,
            dropout_29,
            dropout_3,
            dropout_30,
            dropout_31,
            dropout_4,
            dropout_5,
            dropout_6,
            dropout_7,
            dropout_8,
            dropout_9,
            full_8,
            full_9,
            full_int_array_10,
            full_int_array_2,
            full_int_array_4,
            full_int_array_5,
            full_int_array_8,
            full_int_array_9,
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
            layer_norm_22,
            layer_norm_23,
            layer_norm_3,
            layer_norm_4,
            layer_norm_5,
            layer_norm_6,
            layer_norm_7,
            layer_norm_8,
            layer_norm_9,
            matmul_10,
            matmul_11,
            matmul_12,
            matmul_15,
            matmul_16,
            matmul_17,
            matmul_18,
            matmul_19,
            matmul_2,
            matmul_20,
            matmul_23,
            matmul_24,
            matmul_25,
            matmul_26,
            matmul_27,
            matmul_28,
            matmul_3,
            matmul_31,
            matmul_32,
            matmul_33,
            matmul_4,
            matmul_7,
            matmul_8,
            matmul_9,
            nearest_interp_0,
            nearest_interp_1,
            pool2d_0,
            pool2d_1,
            pool2d_2,
            reshape_11,
            reshape_15,
            reshape_16,
            reshape_3,
            reshape_7,
            slice_0,
            slice_1,
            slice_10,
            slice_11,
            slice_12,
            slice_13,
            slice_14,
            slice_15,
            slice_16,
            slice_17,
            slice_18,
            slice_19,
            slice_2,
            slice_20,
            slice_21,
            slice_22,
            slice_23,
            slice_3,
            slice_4,
            slice_5,
            slice_6,
            slice_7,
            slice_8,
            slice_9,
            softmax_0,
            softmax_1,
            softmax_2,
            softmax_3,
            swish_1,
            swish_10,
            swish_11,
            swish_12,
            swish_13,
            swish_14,
            swish_15,
            swish_16,
            swish_17,
            swish_18,
            swish_19,
            swish_2,
            swish_20,
            swish_21,
            swish_22,
            swish_23,
            swish_24,
            swish_25,
            swish_26,
            swish_27,
            swish_28,
            swish_29,
            swish_3,
            swish_4,
            swish_5,
            swish_6,
            swish_7,
            swish_8,
            swish_9,
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
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_3,
        )

        return swish_0
