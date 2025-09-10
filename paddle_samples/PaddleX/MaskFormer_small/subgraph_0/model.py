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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.conv2d: (1x256x16x16xf32) <- (1x768x16x16xf32, 256x768x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_4, parameter_190, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_190

        # pd_op.group_norm: (1x256x16x16xf32, 1x32xf32, 1x32xf32) <- (1x256x16x16xf32, 256xf32, 256xf32)
        group_norm_0, group_norm_1, group_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_0, parameter_189, parameter_188, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_0, parameter_188, parameter_189

        # pd_op.relu: (1x256x16x16xf32) <- (1x256x16x16xf32)
        relu_0 = paddle._C_ops.relu(group_norm_0)
        del group_norm_0

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x384x32x32xf32, 256x384x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            data_3, parameter_187, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del data_3, parameter_187

        # pd_op.group_norm: (1x256x32x32xf32, 1x32xf32, 1x32xf32) <- (1x256x32x32xf32, 256xf32, 256xf32)
        group_norm_3, group_norm_4, group_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_1, parameter_186, parameter_185, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_1, parameter_185, parameter_186

        # pd_op.nearest_interp: (1x256x32x32xf32) <- (1x256x16x16xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            relu_0, None, None, None, "NCHW", -1, 32, 32, [], "nearest", False, 0
        )
        del relu_0

        # pd_op.add: (1x256x32x32xf32) <- (1x256x32x32xf32, 1x256x32x32xf32)
        add_0 = paddle._C_ops.add(group_norm_3, nearest_interp_0)
        del group_norm_3, nearest_interp_0

        # pd_op.conv2d: (1x256x32x32xf32) <- (1x256x32x32xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_0, parameter_184, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_0, parameter_184

        # pd_op.group_norm: (1x256x32x32xf32, 1x32xf32, 1x32xf32) <- (1x256x32x32xf32, 256xf32, 256xf32)
        group_norm_6, group_norm_7, group_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_2, parameter_183, parameter_182, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_2, parameter_182, parameter_183

        # pd_op.relu: (1x256x32x32xf32) <- (1x256x32x32xf32)
        relu_1 = paddle._C_ops.relu(group_norm_6)
        del group_norm_6

        # pd_op.conv2d: (1x256x64x64xf32) <- (1x192x64x64xf32, 256x192x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            data_2, parameter_181, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del data_2, parameter_181

        # pd_op.group_norm: (1x256x64x64xf32, 1x32xf32, 1x32xf32) <- (1x256x64x64xf32, 256xf32, 256xf32)
        group_norm_9, group_norm_10, group_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_3, parameter_180, parameter_179, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_3, parameter_179, parameter_180

        # pd_op.nearest_interp: (1x256x64x64xf32) <- (1x256x32x32xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            relu_1, None, None, None, "NCHW", -1, 64, 64, [], "nearest", False, 0
        )
        del relu_1

        # pd_op.add: (1x256x64x64xf32) <- (1x256x64x64xf32, 1x256x64x64xf32)
        add_1 = paddle._C_ops.add(group_norm_9, nearest_interp_1)
        del group_norm_9, nearest_interp_1

        # pd_op.conv2d: (1x256x64x64xf32) <- (1x256x64x64xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_1, parameter_178, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_1, parameter_178

        # pd_op.group_norm: (1x256x64x64xf32, 1x32xf32, 1x32xf32) <- (1x256x64x64xf32, 256xf32, 256xf32)
        group_norm_12, group_norm_13, group_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_4, parameter_177, parameter_176, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_4, parameter_176, parameter_177

        # pd_op.relu: (1x256x64x64xf32) <- (1x256x64x64xf32)
        relu_2 = paddle._C_ops.relu(group_norm_12)
        del group_norm_12

        # pd_op.conv2d: (1x256x128x128xf32) <- (1x96x128x128xf32, 256x96x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            data_1, parameter_175, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del data_1, parameter_175

        # pd_op.group_norm: (1x256x128x128xf32, 1x32xf32, 1x32xf32) <- (1x256x128x128xf32, 256xf32, 256xf32)
        group_norm_15, group_norm_16, group_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_5, parameter_174, parameter_173, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_5, parameter_173, parameter_174

        # pd_op.nearest_interp: (1x256x128x128xf32) <- (1x256x64x64xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(
            relu_2, None, None, None, "NCHW", -1, 128, 128, [], "nearest", False, 0
        )
        del relu_2

        # pd_op.add: (1x256x128x128xf32) <- (1x256x128x128xf32, 1x256x128x128xf32)
        add_2 = paddle._C_ops.add(group_norm_15, nearest_interp_2)
        del group_norm_15, nearest_interp_2

        # pd_op.conv2d: (1x256x128x128xf32) <- (1x256x128x128xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            add_2, parameter_172, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_2, parameter_172

        # pd_op.group_norm: (1x256x128x128xf32, 1x32xf32, 1x32xf32) <- (1x256x128x128xf32, 256xf32, 256xf32)
        group_norm_18, group_norm_19, group_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.group_norm(
                conv2d_6, parameter_171, parameter_170, float("1e-05"), 32, "NCHW"
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del conv2d_6, parameter_170, parameter_171

        # pd_op.relu: (1x256x128x128xf32) <- (1x256x128x128xf32)
        relu_3 = paddle._C_ops.relu(group_norm_18)
        del group_norm_18

        # pd_op.conv2d: (1x256x128x128xf32) <- (1x256x128x128xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_3, parameter_169, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_169, relu_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_168, full_int_array_0)
        del parameter_168

        # pd_op.add: (1x256x128x128xf32) <- (1x256x128x128xf32, 1x256x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_7, reshape_0)
        del conv2d_7, reshape_0

        # pd_op.full: (1x16x16xb) <- ()
        full_0 = paddle._C_ops.full(
            [1, 16, 16],
            float("0"),
            paddle.bool,
            paddle.framework._current_expected_place(),
        )

        # pd_op.bitwise_not: (1x16x16xb) <- (1x16x16xb)
        bitwise_not_0 = paddle._C_ops.bitwise_not(full_0)
        del full_0

        # pd_op.cast: (1x16x16xf32) <- (1x16x16xb)
        cast_0 = paddle._C_ops.cast(bitwise_not_0, paddle.float32)
        del bitwise_not_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x16x16xf32) <- (1x16x16xf32, 1xi32)
        cumsum_0 = paddle._C_ops.cumsum(cast_0, full_1, False, False, False)
        del full_1

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.cumsum: (1x16x16xf32) <- (1x16x16xf32, 1xi32)
        cumsum_1 = paddle._C_ops.cumsum(cast_0, full_2, False, False, False)
        del cast_0, full_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2147483647]

        # pd_op.slice: (1x1x16xf32) <- (1x16x16xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            cumsum_0, [1], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x1x16xf32) <- (1x1x16xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(slice_0, full_3, float("1e-06"), True)
        del slice_0

        # pd_op.divide: (1x16x16xf32) <- (1x16x16xf32, 1x1x16xf32)
        divide_0 = paddle._C_ops.divide(cumsum_0, scale_0)
        del cumsum_0, scale_0

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("6.28319"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x16x16xf32) <- (1x16x16xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(divide_0, full_4, float("0"), True)
        del divide_0

        # pd_op.slice: (1x16x1xf32) <- (1x16x16xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            cumsum_1, [2], full_int_array_1, full_int_array_2, [1], []
        )

        # pd_op.scale: (1x16x1xf32) <- (1x16x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_1, full_3, float("1e-06"), True)
        del slice_1

        # pd_op.divide: (1x16x16xf32) <- (1x16x16xf32, 1x16x1xf32)
        divide_1 = paddle._C_ops.divide(cumsum_1, scale_2)
        del cumsum_1, scale_2

        # pd_op.scale: (1x16x16xf32) <- (1x16x16xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(divide_1, full_4, float("0"), True)
        del divide_1, full_4

        # pd_op.full: (1xf64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("128"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (128xf32) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_5, full_6, full_7, dtype="float32")
        del full_5, full_6, full_7

        # pd_op.cast: (128xi64) <- (128xf32)
        cast_1 = paddle._C_ops.cast(arange_0, paddle.int64)
        del arange_0

        # pd_op.full_like: (128xi64) <- (128xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            cast_1, full_3, paddle.int64, paddle.framework._current_expected_place()
        )
        del full_3

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xi64) <- (128xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(full_like_0, full_8, float("0"), True)
        del full_like_0

        # pd_op.floor_divide: (128xi64) <- (128xi64, 128xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(cast_1, scale_4)
        del cast_1, scale_4

        # pd_op.scale: (128xi64) <- (128xi64, 1xf32)
        scale_5 = paddle._C_ops.scale(floor_divide_0, full_8, float("0"), True)
        del floor_divide_0, full_8

        # pd_op.cast: (128xf32) <- (128xi64)
        cast_2 = paddle._C_ops.cast(scale_5, paddle.float32)
        del scale_5

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("0.0078125"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (128xf32) <- (128xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_2, full_9, float("0"), True)
        del cast_2, full_9

        # pd_op.full: (128xf32) <- ()
        full_10 = paddle._C_ops.full(
            [128],
            float("10000"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.elementwise_pow: (128xf32) <- (128xf32, 128xf32)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(full_10, scale_6)
        del full_10, scale_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.unsqueeze: (1x16x16x1xf32) <- (1x16x16xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_3, full_int_array_3)
        del scale_3

        # pd_op.divide: (1x16x16x128xf32) <- (1x16x16x1xf32, 128xf32)
        divide_2 = paddle._C_ops.divide(unsqueeze_1, elementwise_pow_0)
        del unsqueeze_1

        # pd_op.unsqueeze: (1x16x16x1xf32) <- (1x16x16xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(scale_1, full_int_array_3)
        del scale_1

        # pd_op.divide: (1x16x16x128xf32) <- (1x16x16x1xf32, 128xf32)
        divide_3 = paddle._C_ops.divide(unsqueeze_2, elementwise_pow_0)
        del elementwise_pow_0, unsqueeze_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.strided_slice: (1x16x16x64xf32) <- (1x16x16x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_0 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_4, full_int_array_2, full_int_array_5
        )

        # pd_op.sin: (1x16x16x64xf32) <- (1x16x16x64xf32)
        sin_0 = paddle._C_ops.sin(strided_slice_0)
        del strided_slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.strided_slice: (1x16x16x64xf32) <- (1x16x16x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_1 = paddle._C_ops.strided_slice(
            divide_2, [3], full_int_array_6, full_int_array_2, full_int_array_5
        )
        del divide_2

        # pd_op.cos: (1x16x16x64xf32) <- (1x16x16x64xf32)
        cos_0 = paddle._C_ops.cos(strided_slice_1)
        del strided_slice_1

        # builtin.combine: ([1x16x16x64xf32, 1x16x16x64xf32]) <- (1x16x16x64xf32, 1x16x16x64xf32)
        combine_0 = [sin_0, cos_0]
        del cos_0, sin_0

        # pd_op.stack: (1x16x16x64x2xf32) <- ([1x16x16x64xf32, 1x16x16x64xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 4)
        del combine_0

        # pd_op.flatten: (1x16x16x128xf32) <- (1x16x16x64x2xf32)
        flatten_0 = paddle._C_ops.flatten(stack_0, 3, 4)
        del stack_0

        # pd_op.strided_slice: (1x16x16x64xf32) <- (1x16x16x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_2 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_4, full_int_array_2, full_int_array_5
        )

        # pd_op.sin: (1x16x16x64xf32) <- (1x16x16x64xf32)
        sin_1 = paddle._C_ops.sin(strided_slice_2)
        del strided_slice_2

        # pd_op.strided_slice: (1x16x16x64xf32) <- (1x16x16x128xf32, 1xi64, 1xi64, 1xi64)
        strided_slice_3 = paddle._C_ops.strided_slice(
            divide_3, [3], full_int_array_6, full_int_array_2, full_int_array_5
        )
        del divide_3

        # pd_op.cos: (1x16x16x64xf32) <- (1x16x16x64xf32)
        cos_1 = paddle._C_ops.cos(strided_slice_3)
        del strided_slice_3

        # builtin.combine: ([1x16x16x64xf32, 1x16x16x64xf32]) <- (1x16x16x64xf32, 1x16x16x64xf32)
        combine_1 = [sin_1, cos_1]
        del cos_1, sin_1

        # pd_op.stack: (1x16x16x64x2xf32) <- ([1x16x16x64xf32, 1x16x16x64xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 4)
        del combine_1

        # pd_op.flatten: (1x16x16x128xf32) <- (1x16x16x64x2xf32)
        flatten_1 = paddle._C_ops.flatten(stack_1, 3, 4)
        del stack_1

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full(
            [1], float("3"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x16x16x128xf32, 1x16x16x128xf32]) <- (1x16x16x128xf32, 1x16x16x128xf32)
        combine_2 = [flatten_1, flatten_0]
        del flatten_0, flatten_1

        # pd_op.concat: (1x16x16x256xf32) <- ([1x16x16x128xf32, 1x16x16x128xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_11)
        del combine_2, full_11

        # pd_op.transpose: (1x256x16x16xf32) <- (1x16x16x256xf32)
        transpose_0 = paddle._C_ops.transpose(concat_0, [0, 3, 1, 2])
        del concat_0

        # pd_op.conv2d: (1x256x16x16xf32) <- (1x768x16x16xf32, 256x768x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            data_4, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_4, parameter_167

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_166, full_int_array_0)
        del full_int_array_0, parameter_166

        # pd_op.add: (1x256x16x16xf32) <- (1x256x16x16xf32, 1x256x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_8, reshape_1)
        del conv2d_8, reshape_1

        # pd_op.flatten: (1x256x256xf32) <- (1x256x16x16xf32)
        flatten_2 = paddle._C_ops.flatten(add_4, 2, 3)
        del add_4

        # pd_op.transpose: (256x1x256xf32) <- (1x256x256xf32)
        transpose_1 = paddle._C_ops.transpose(flatten_2, [2, 0, 1])
        del flatten_2

        # pd_op.flatten: (1x256x256xf32) <- (1x256x16x16xf32)
        flatten_3 = paddle._C_ops.flatten(transpose_0, 2, 3)
        del transpose_0

        # pd_op.transpose: (256x1x256xf32) <- (1x256x256xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_3, [2, 0, 1])
        del flatten_3

        # builtin.combine: ([100x256xf32]) <- (100x256xf32)
        combine_3 = [data_0]
        del data_0

        # pd_op.stack: (100x1x256xf32) <- ([100x256xf32])
        stack_2 = paddle._C_ops.stack(combine_3, 1)
        del combine_3

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (100x1x256xf32) <- (100x1x256xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            stack_2, full_12, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_12

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_5 = paddle._C_ops.add(full_like_1, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_3 = paddle._C_ops.transpose(add_5, [1, 0, 2])
        del add_5

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_4 = paddle._C_ops.transpose(full_like_1, [1, 0, 2])
        del full_like_1

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(transpose_3, parameter_165, False, False)
        del parameter_165

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_6 = paddle._C_ops.add(matmul_0, parameter_164)
        del matmul_0, parameter_164

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [0, 0, 8, 32]

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_6, full_int_array_7)
        del add_6

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_5 = paddle._C_ops.transpose(reshape_2, [0, 2, 1, 3])
        del reshape_2

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_1 = paddle._C_ops.matmul(transpose_3, parameter_163, False, False)
        del parameter_163, transpose_3

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_7 = paddle._C_ops.add(matmul_1, parameter_162)
        del matmul_1, parameter_162

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_2 = paddle._C_ops.matmul(transpose_4, parameter_161, False, False)
        del parameter_161

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_8 = paddle._C_ops.add(matmul_2, parameter_160)
        del matmul_2, parameter_160

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(add_7, full_int_array_7)
        del add_7

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_6 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3])
        del reshape_3

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(add_8, full_int_array_7)
        del add_8

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [0, 2, 1, 3])
        del reshape_4

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("0.176777"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(transpose_5, full_13, float("0"), True)
        del transpose_5

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_3 = paddle._C_ops.matmul(scale_7, transpose_6, False, True)
        del scale_7, transpose_6

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_0 = paddle._C_ops.softmax(matmul_3, -1)
        del matmul_3

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_0, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_0

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_4 = paddle._C_ops.matmul(dropout_0, transpose_7, False, False)
        del dropout_0, transpose_7

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_8 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])
        del matmul_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [0, 0, 256]

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(transpose_8, full_int_array_8)
        del transpose_8

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_5 = paddle._C_ops.matmul(reshape_5, parameter_159, False, False)
        del parameter_159, reshape_5

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_9 = paddle._C_ops.add(matmul_5, parameter_158)
        del matmul_5, parameter_158

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_9 = paddle._C_ops.transpose(add_9, [1, 0, 2])
        del add_9

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_10 = paddle._C_ops.transpose(transpose_4, [1, 0, 2])
        del transpose_4

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_9, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_9

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_10 = paddle._C_ops.add(transpose_10, dropout_2)
        del dropout_2, transpose_10

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_10, parameter_157, parameter_156, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_10, parameter_156, parameter_157

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_11 = paddle._C_ops.add(layer_norm_0, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_11 = paddle._C_ops.transpose(add_11, [1, 0, 2])
        del add_11

        # pd_op.add: (256x1x256xf32) <- (256x1x256xf32, 256x1x256xf32)
        add_12 = paddle._C_ops.add(transpose_1, transpose_2)
        del transpose_2

        # pd_op.transpose: (1x256x256xf32) <- (256x1x256xf32)
        transpose_12 = paddle._C_ops.transpose(add_12, [1, 0, 2])
        del add_12

        # pd_op.transpose: (1x256x256xf32) <- (256x1x256xf32)
        transpose_13 = paddle._C_ops.transpose(transpose_1, [1, 0, 2])

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_6 = paddle._C_ops.matmul(transpose_11, parameter_155, False, False)
        del parameter_155, transpose_11

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_13 = paddle._C_ops.add(matmul_6, parameter_154)
        del matmul_6, parameter_154

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(add_13, full_int_array_7)
        del add_13

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_14 = paddle._C_ops.transpose(reshape_6, [0, 2, 1, 3])
        del reshape_6

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_7 = paddle._C_ops.matmul(transpose_12, parameter_153, False, False)
        del parameter_153

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_14 = paddle._C_ops.add(matmul_7, parameter_152)
        del matmul_7, parameter_152

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_8 = paddle._C_ops.matmul(transpose_13, parameter_151, False, False)
        del parameter_151

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_15 = paddle._C_ops.add(matmul_8, parameter_150)
        del matmul_8, parameter_150

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(add_14, full_int_array_7)
        del add_14

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_15 = paddle._C_ops.transpose(reshape_7, [0, 2, 1, 3])
        del reshape_7

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(add_15, full_int_array_7)
        del add_15

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_16 = paddle._C_ops.transpose(reshape_8, [0, 2, 1, 3])
        del reshape_8

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(transpose_14, full_13, float("0"), True)
        del transpose_14

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_9 = paddle._C_ops.matmul(scale_8, transpose_15, False, True)
        del scale_8, transpose_15

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_1 = paddle._C_ops.softmax(matmul_9, -1)
        del matmul_9

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_1, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_1

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_10 = paddle._C_ops.matmul(dropout_4, transpose_16, False, False)
        del dropout_4, transpose_16

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])
        del matmul_10

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_17, full_int_array_8)
        del transpose_17

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_11 = paddle._C_ops.matmul(reshape_9, parameter_149, False, False)
        del parameter_149, reshape_9

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_16 = paddle._C_ops.add(matmul_11, parameter_148)
        del matmul_11, parameter_148

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_18 = paddle._C_ops.transpose(add_16, [1, 0, 2])
        del add_16

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_18, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_18

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_17 = paddle._C_ops.add(layer_norm_0, dropout_6)
        del dropout_6, layer_norm_0

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_17, parameter_147, parameter_146, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_17, parameter_146, parameter_147

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_3, parameter_145, False, False)
        del parameter_145

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_18 = paddle._C_ops.add(matmul_12, parameter_144)
        del matmul_12, parameter_144

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_4 = paddle._C_ops.relu(add_18)
        del add_18

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_4, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_4

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_13 = paddle._C_ops.matmul(dropout_8, parameter_143, False, False)
        del dropout_8, parameter_143

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_19 = paddle._C_ops.add(matmul_13, parameter_142)
        del matmul_13, parameter_142

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_19, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_19

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_20 = paddle._C_ops.add(layer_norm_3, dropout_10)
        del dropout_10, layer_norm_3

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_20, parameter_141, parameter_140, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_20, parameter_140, parameter_141

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_6, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_21 = paddle._C_ops.add(layer_norm_6, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_19 = paddle._C_ops.transpose(add_21, [1, 0, 2])
        del add_21

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_20 = paddle._C_ops.transpose(layer_norm_6, [1, 0, 2])
        del layer_norm_6

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_14 = paddle._C_ops.matmul(transpose_19, parameter_137, False, False)
        del parameter_137

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_22 = paddle._C_ops.add(matmul_14, parameter_136)
        del matmul_14, parameter_136

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(add_22, full_int_array_7)
        del add_22

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_21 = paddle._C_ops.transpose(reshape_10, [0, 2, 1, 3])
        del reshape_10

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_15 = paddle._C_ops.matmul(transpose_19, parameter_135, False, False)
        del parameter_135, transpose_19

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_23 = paddle._C_ops.add(matmul_15, parameter_134)
        del matmul_15, parameter_134

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_16 = paddle._C_ops.matmul(transpose_20, parameter_133, False, False)
        del parameter_133

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_24 = paddle._C_ops.add(matmul_16, parameter_132)
        del matmul_16, parameter_132

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(add_23, full_int_array_7)
        del add_23

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_22 = paddle._C_ops.transpose(reshape_11, [0, 2, 1, 3])
        del reshape_11

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(add_24, full_int_array_7)
        del add_24

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_23 = paddle._C_ops.transpose(reshape_12, [0, 2, 1, 3])
        del reshape_12

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(transpose_21, full_13, float("0"), True)
        del transpose_21

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_17 = paddle._C_ops.matmul(scale_9, transpose_22, False, True)
        del scale_9, transpose_22

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_2 = paddle._C_ops.softmax(matmul_17, -1)
        del matmul_17

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_2, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_2

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_18 = paddle._C_ops.matmul(dropout_12, transpose_23, False, False)
        del dropout_12, transpose_23

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_24 = paddle._C_ops.transpose(matmul_18, [0, 2, 1, 3])
        del matmul_18

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_13 = paddle._C_ops.reshape(transpose_24, full_int_array_8)
        del transpose_24

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_19 = paddle._C_ops.matmul(reshape_13, parameter_131, False, False)
        del parameter_131, reshape_13

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_25 = paddle._C_ops.add(matmul_19, parameter_130)
        del matmul_19, parameter_130

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_25 = paddle._C_ops.transpose(add_25, [1, 0, 2])
        del add_25

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_26 = paddle._C_ops.transpose(transpose_20, [1, 0, 2])
        del transpose_20

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_25, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_25

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_26 = paddle._C_ops.add(transpose_26, dropout_14)
        del dropout_14, transpose_26

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_26, parameter_129, parameter_128, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_26, parameter_128, parameter_129

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_27 = paddle._C_ops.add(layer_norm_12, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_27 = paddle._C_ops.transpose(add_27, [1, 0, 2])
        del add_27

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_20 = paddle._C_ops.matmul(transpose_27, parameter_127, False, False)
        del parameter_127, transpose_27

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_28 = paddle._C_ops.add(matmul_20, parameter_126)
        del matmul_20, parameter_126

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(add_28, full_int_array_7)
        del add_28

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_28 = paddle._C_ops.transpose(reshape_14, [0, 2, 1, 3])
        del reshape_14

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_21 = paddle._C_ops.matmul(transpose_12, parameter_125, False, False)
        del parameter_125

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_29 = paddle._C_ops.add(matmul_21, parameter_124)
        del matmul_21, parameter_124

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_22 = paddle._C_ops.matmul(transpose_13, parameter_123, False, False)
        del parameter_123

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_30 = paddle._C_ops.add(matmul_22, parameter_122)
        del matmul_22, parameter_122

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(add_29, full_int_array_7)
        del add_29

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_29 = paddle._C_ops.transpose(reshape_15, [0, 2, 1, 3])
        del reshape_15

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(add_30, full_int_array_7)
        del add_30

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_30 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3])
        del reshape_16

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(transpose_28, full_13, float("0"), True)
        del transpose_28

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_23 = paddle._C_ops.matmul(scale_10, transpose_29, False, True)
        del scale_10, transpose_29

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_3 = paddle._C_ops.softmax(matmul_23, -1)
        del matmul_23

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_3, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_3

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_24 = paddle._C_ops.matmul(dropout_16, transpose_30, False, False)
        del dropout_16, transpose_30

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_31 = paddle._C_ops.transpose(matmul_24, [0, 2, 1, 3])
        del matmul_24

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_17 = paddle._C_ops.reshape(transpose_31, full_int_array_8)
        del transpose_31

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_25 = paddle._C_ops.matmul(reshape_17, parameter_121, False, False)
        del parameter_121, reshape_17

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_31 = paddle._C_ops.add(matmul_25, parameter_120)
        del matmul_25, parameter_120

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_32 = paddle._C_ops.transpose(add_31, [1, 0, 2])
        del add_31

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_32, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_32

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_32 = paddle._C_ops.add(layer_norm_12, dropout_18)
        del dropout_18, layer_norm_12

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_32, parameter_119, parameter_118, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_32, parameter_118, parameter_119

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_26 = paddle._C_ops.matmul(layer_norm_15, parameter_117, False, False)
        del parameter_117

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_33 = paddle._C_ops.add(matmul_26, parameter_116)
        del matmul_26, parameter_116

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_5 = paddle._C_ops.relu(add_33)
        del add_33

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_5, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_5

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_27 = paddle._C_ops.matmul(dropout_20, parameter_115, False, False)
        del dropout_20, parameter_115

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_34 = paddle._C_ops.add(matmul_27, parameter_114)
        del matmul_27, parameter_114

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_34, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_34

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_35 = paddle._C_ops.add(layer_norm_15, dropout_22)
        del dropout_22, layer_norm_15

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_35, parameter_113, parameter_112, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_35, parameter_112, parameter_113

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_18, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_36 = paddle._C_ops.add(layer_norm_18, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_33 = paddle._C_ops.transpose(add_36, [1, 0, 2])
        del add_36

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_34 = paddle._C_ops.transpose(layer_norm_18, [1, 0, 2])
        del layer_norm_18

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_28 = paddle._C_ops.matmul(transpose_33, parameter_111, False, False)
        del parameter_111

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_37 = paddle._C_ops.add(matmul_28, parameter_110)
        del matmul_28, parameter_110

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(add_37, full_int_array_7)
        del add_37

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_35 = paddle._C_ops.transpose(reshape_18, [0, 2, 1, 3])
        del reshape_18

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_29 = paddle._C_ops.matmul(transpose_33, parameter_109, False, False)
        del parameter_109, transpose_33

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_38 = paddle._C_ops.add(matmul_29, parameter_108)
        del matmul_29, parameter_108

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_30 = paddle._C_ops.matmul(transpose_34, parameter_107, False, False)
        del parameter_107

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_39 = paddle._C_ops.add(matmul_30, parameter_106)
        del matmul_30, parameter_106

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(add_38, full_int_array_7)
        del add_38

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_36 = paddle._C_ops.transpose(reshape_19, [0, 2, 1, 3])
        del reshape_19

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(add_39, full_int_array_7)
        del add_39

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_37 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3])
        del reshape_20

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(transpose_35, full_13, float("0"), True)
        del transpose_35

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_31 = paddle._C_ops.matmul(scale_11, transpose_36, False, True)
        del scale_11, transpose_36

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_4 = paddle._C_ops.softmax(matmul_31, -1)
        del matmul_31

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_4, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_4

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_32 = paddle._C_ops.matmul(dropout_24, transpose_37, False, False)
        del dropout_24, transpose_37

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_38 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])
        del matmul_32

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_21 = paddle._C_ops.reshape(transpose_38, full_int_array_8)
        del transpose_38

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_33 = paddle._C_ops.matmul(reshape_21, parameter_105, False, False)
        del parameter_105, reshape_21

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_40 = paddle._C_ops.add(matmul_33, parameter_104)
        del matmul_33, parameter_104

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_39 = paddle._C_ops.transpose(add_40, [1, 0, 2])
        del add_40

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_40 = paddle._C_ops.transpose(transpose_34, [1, 0, 2])
        del transpose_34

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_39, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_39

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_41 = paddle._C_ops.add(transpose_40, dropout_26)
        del dropout_26, transpose_40

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_41, parameter_103, parameter_102, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_41, parameter_102, parameter_103

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_42 = paddle._C_ops.add(layer_norm_24, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_41 = paddle._C_ops.transpose(add_42, [1, 0, 2])
        del add_42

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_34 = paddle._C_ops.matmul(transpose_41, parameter_101, False, False)
        del parameter_101, transpose_41

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_43 = paddle._C_ops.add(matmul_34, parameter_100)
        del matmul_34, parameter_100

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(add_43, full_int_array_7)
        del add_43

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_42 = paddle._C_ops.transpose(reshape_22, [0, 2, 1, 3])
        del reshape_22

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_35 = paddle._C_ops.matmul(transpose_12, parameter_99, False, False)
        del parameter_99

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_44 = paddle._C_ops.add(matmul_35, parameter_98)
        del matmul_35, parameter_98

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_36 = paddle._C_ops.matmul(transpose_13, parameter_97, False, False)
        del parameter_97

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_45 = paddle._C_ops.add(matmul_36, parameter_96)
        del matmul_36, parameter_96

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(add_44, full_int_array_7)
        del add_44

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_43 = paddle._C_ops.transpose(reshape_23, [0, 2, 1, 3])
        del reshape_23

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(add_45, full_int_array_7)
        del add_45

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_44 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3])
        del reshape_24

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(transpose_42, full_13, float("0"), True)
        del transpose_42

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_37 = paddle._C_ops.matmul(scale_12, transpose_43, False, True)
        del scale_12, transpose_43

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_5 = paddle._C_ops.softmax(matmul_37, -1)
        del matmul_37

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_5, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_5

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_38 = paddle._C_ops.matmul(dropout_28, transpose_44, False, False)
        del dropout_28, transpose_44

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])
        del matmul_38

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_25 = paddle._C_ops.reshape(transpose_45, full_int_array_8)
        del transpose_45

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_39 = paddle._C_ops.matmul(reshape_25, parameter_95, False, False)
        del parameter_95, reshape_25

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_46 = paddle._C_ops.add(matmul_39, parameter_94)
        del matmul_39, parameter_94

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_46 = paddle._C_ops.transpose(add_46, [1, 0, 2])
        del add_46

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_46, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_46

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_47 = paddle._C_ops.add(layer_norm_24, dropout_30)
        del dropout_30, layer_norm_24

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_47, parameter_93, parameter_92, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_47, parameter_92, parameter_93

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_27, parameter_91, False, False)
        del parameter_91

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_48 = paddle._C_ops.add(matmul_40, parameter_90)
        del matmul_40, parameter_90

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_6 = paddle._C_ops.relu(add_48)
        del add_48

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_6, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_6

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_41 = paddle._C_ops.matmul(dropout_32, parameter_89, False, False)
        del dropout_32, parameter_89

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_49 = paddle._C_ops.add(matmul_41, parameter_88)
        del matmul_41, parameter_88

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_49, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_49

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_50 = paddle._C_ops.add(layer_norm_27, dropout_34)
        del dropout_34, layer_norm_27

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_50, parameter_87, parameter_86, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_50, parameter_86, parameter_87

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_30, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_51 = paddle._C_ops.add(layer_norm_30, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_47 = paddle._C_ops.transpose(add_51, [1, 0, 2])
        del add_51

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_48 = paddle._C_ops.transpose(layer_norm_30, [1, 0, 2])
        del layer_norm_30

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_42 = paddle._C_ops.matmul(transpose_47, parameter_85, False, False)
        del parameter_85

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_52 = paddle._C_ops.add(matmul_42, parameter_84)
        del matmul_42, parameter_84

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(add_52, full_int_array_7)
        del add_52

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_49 = paddle._C_ops.transpose(reshape_26, [0, 2, 1, 3])
        del reshape_26

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_43 = paddle._C_ops.matmul(transpose_47, parameter_83, False, False)
        del parameter_83, transpose_47

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_53 = paddle._C_ops.add(matmul_43, parameter_82)
        del matmul_43, parameter_82

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_44 = paddle._C_ops.matmul(transpose_48, parameter_81, False, False)
        del parameter_81

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_54 = paddle._C_ops.add(matmul_44, parameter_80)
        del matmul_44, parameter_80

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(add_53, full_int_array_7)
        del add_53

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_50 = paddle._C_ops.transpose(reshape_27, [0, 2, 1, 3])
        del reshape_27

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(add_54, full_int_array_7)
        del add_54

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_51 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3])
        del reshape_28

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(transpose_49, full_13, float("0"), True)
        del transpose_49

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_45 = paddle._C_ops.matmul(scale_13, transpose_50, False, True)
        del scale_13, transpose_50

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_6 = paddle._C_ops.softmax(matmul_45, -1)
        del matmul_45

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_6, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_6

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_46 = paddle._C_ops.matmul(dropout_36, transpose_51, False, False)
        del dropout_36, transpose_51

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_52 = paddle._C_ops.transpose(matmul_46, [0, 2, 1, 3])
        del matmul_46

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_29 = paddle._C_ops.reshape(transpose_52, full_int_array_8)
        del transpose_52

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_47 = paddle._C_ops.matmul(reshape_29, parameter_79, False, False)
        del parameter_79, reshape_29

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_55 = paddle._C_ops.add(matmul_47, parameter_78)
        del matmul_47, parameter_78

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_53 = paddle._C_ops.transpose(add_55, [1, 0, 2])
        del add_55

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_54 = paddle._C_ops.transpose(transpose_48, [1, 0, 2])
        del transpose_48

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_53, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_53

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_56 = paddle._C_ops.add(transpose_54, dropout_38)
        del dropout_38, transpose_54

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_56, parameter_77, parameter_76, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_56, parameter_76, parameter_77

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_57 = paddle._C_ops.add(layer_norm_36, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_55 = paddle._C_ops.transpose(add_57, [1, 0, 2])
        del add_57

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_48 = paddle._C_ops.matmul(transpose_55, parameter_75, False, False)
        del parameter_75, transpose_55

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_58 = paddle._C_ops.add(matmul_48, parameter_74)
        del matmul_48, parameter_74

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(add_58, full_int_array_7)
        del add_58

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_56 = paddle._C_ops.transpose(reshape_30, [0, 2, 1, 3])
        del reshape_30

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_49 = paddle._C_ops.matmul(transpose_12, parameter_73, False, False)
        del parameter_73

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_59 = paddle._C_ops.add(matmul_49, parameter_72)
        del matmul_49, parameter_72

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_50 = paddle._C_ops.matmul(transpose_13, parameter_71, False, False)
        del parameter_71

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_60 = paddle._C_ops.add(matmul_50, parameter_70)
        del matmul_50, parameter_70

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(add_59, full_int_array_7)
        del add_59

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_57 = paddle._C_ops.transpose(reshape_31, [0, 2, 1, 3])
        del reshape_31

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(add_60, full_int_array_7)
        del add_60

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_58 = paddle._C_ops.transpose(reshape_32, [0, 2, 1, 3])
        del reshape_32

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(transpose_56, full_13, float("0"), True)
        del transpose_56

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_51 = paddle._C_ops.matmul(scale_14, transpose_57, False, True)
        del scale_14, transpose_57

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_7 = paddle._C_ops.softmax(matmul_51, -1)
        del matmul_51

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_7, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_7

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_52 = paddle._C_ops.matmul(dropout_40, transpose_58, False, False)
        del dropout_40, transpose_58

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_59 = paddle._C_ops.transpose(matmul_52, [0, 2, 1, 3])
        del matmul_52

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_33 = paddle._C_ops.reshape(transpose_59, full_int_array_8)
        del transpose_59

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_53 = paddle._C_ops.matmul(reshape_33, parameter_69, False, False)
        del parameter_69, reshape_33

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_61 = paddle._C_ops.add(matmul_53, parameter_68)
        del matmul_53, parameter_68

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_60 = paddle._C_ops.transpose(add_61, [1, 0, 2])
        del add_61

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_60, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_60

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_62 = paddle._C_ops.add(layer_norm_36, dropout_42)
        del dropout_42, layer_norm_36

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_62, parameter_67, parameter_66, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_62, parameter_66, parameter_67

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_39, parameter_65, False, False)
        del parameter_65

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_63 = paddle._C_ops.add(matmul_54, parameter_64)
        del matmul_54, parameter_64

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_7 = paddle._C_ops.relu(add_63)
        del add_63

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_7, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_7

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_55 = paddle._C_ops.matmul(dropout_44, parameter_63, False, False)
        del dropout_44, parameter_63

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_64 = paddle._C_ops.add(matmul_55, parameter_62)
        del matmul_55, parameter_62

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_64, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_64

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_65 = paddle._C_ops.add(layer_norm_39, dropout_46)
        del dropout_46, layer_norm_39

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_65, parameter_61, parameter_60, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_65, parameter_60, parameter_61

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_42, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_66 = paddle._C_ops.add(layer_norm_42, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_61 = paddle._C_ops.transpose(add_66, [1, 0, 2])
        del add_66

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_62 = paddle._C_ops.transpose(layer_norm_42, [1, 0, 2])
        del layer_norm_42

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_56 = paddle._C_ops.matmul(transpose_61, parameter_59, False, False)
        del parameter_59

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_67 = paddle._C_ops.add(matmul_56, parameter_58)
        del matmul_56, parameter_58

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(add_67, full_int_array_7)
        del add_67

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_63 = paddle._C_ops.transpose(reshape_34, [0, 2, 1, 3])
        del reshape_34

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_57 = paddle._C_ops.matmul(transpose_61, parameter_57, False, False)
        del parameter_57, transpose_61

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_68 = paddle._C_ops.add(matmul_57, parameter_56)
        del matmul_57, parameter_56

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_58 = paddle._C_ops.matmul(transpose_62, parameter_55, False, False)
        del parameter_55

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_69 = paddle._C_ops.add(matmul_58, parameter_54)
        del matmul_58, parameter_54

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(add_68, full_int_array_7)
        del add_68

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_64 = paddle._C_ops.transpose(reshape_35, [0, 2, 1, 3])
        del reshape_35

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(add_69, full_int_array_7)
        del add_69

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_65 = paddle._C_ops.transpose(reshape_36, [0, 2, 1, 3])
        del reshape_36

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(transpose_63, full_13, float("0"), True)
        del transpose_63

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_59 = paddle._C_ops.matmul(scale_15, transpose_64, False, True)
        del scale_15, transpose_64

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_8 = paddle._C_ops.softmax(matmul_59, -1)
        del matmul_59

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_8, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_8

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_60 = paddle._C_ops.matmul(dropout_48, transpose_65, False, False)
        del dropout_48, transpose_65

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_66 = paddle._C_ops.transpose(matmul_60, [0, 2, 1, 3])
        del matmul_60

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_37 = paddle._C_ops.reshape(transpose_66, full_int_array_8)
        del transpose_66

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_61 = paddle._C_ops.matmul(reshape_37, parameter_53, False, False)
        del parameter_53, reshape_37

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_70 = paddle._C_ops.add(matmul_61, parameter_52)
        del matmul_61, parameter_52

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_67 = paddle._C_ops.transpose(add_70, [1, 0, 2])
        del add_70

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_68 = paddle._C_ops.transpose(transpose_62, [1, 0, 2])
        del transpose_62

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_67, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_67

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_71 = paddle._C_ops.add(transpose_68, dropout_50)
        del dropout_50, transpose_68

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_71, parameter_51, parameter_50, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_71, parameter_50, parameter_51

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_72 = paddle._C_ops.add(layer_norm_48, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_69 = paddle._C_ops.transpose(add_72, [1, 0, 2])
        del add_72

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_62 = paddle._C_ops.matmul(transpose_69, parameter_49, False, False)
        del parameter_49, transpose_69

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_73 = paddle._C_ops.add(matmul_62, parameter_48)
        del matmul_62, parameter_48

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_38 = paddle._C_ops.reshape(add_73, full_int_array_7)
        del add_73

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_70 = paddle._C_ops.transpose(reshape_38, [0, 2, 1, 3])
        del reshape_38

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_63 = paddle._C_ops.matmul(transpose_12, parameter_47, False, False)
        del parameter_47

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_74 = paddle._C_ops.add(matmul_63, parameter_46)
        del matmul_63, parameter_46

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_64 = paddle._C_ops.matmul(transpose_13, parameter_45, False, False)
        del parameter_45

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_75 = paddle._C_ops.add(matmul_64, parameter_44)
        del matmul_64, parameter_44

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_39 = paddle._C_ops.reshape(add_74, full_int_array_7)
        del add_74

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_71 = paddle._C_ops.transpose(reshape_39, [0, 2, 1, 3])
        del reshape_39

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_40 = paddle._C_ops.reshape(add_75, full_int_array_7)
        del add_75

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_72 = paddle._C_ops.transpose(reshape_40, [0, 2, 1, 3])
        del reshape_40

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(transpose_70, full_13, float("0"), True)
        del transpose_70

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_65 = paddle._C_ops.matmul(scale_16, transpose_71, False, True)
        del scale_16, transpose_71

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_9 = paddle._C_ops.softmax(matmul_65, -1)
        del matmul_65

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_9, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_9

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_66 = paddle._C_ops.matmul(dropout_52, transpose_72, False, False)
        del dropout_52, transpose_72

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_73 = paddle._C_ops.transpose(matmul_66, [0, 2, 1, 3])
        del matmul_66

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_41 = paddle._C_ops.reshape(transpose_73, full_int_array_8)
        del transpose_73

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_67 = paddle._C_ops.matmul(reshape_41, parameter_43, False, False)
        del parameter_43, reshape_41

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_76 = paddle._C_ops.add(matmul_67, parameter_42)
        del matmul_67, parameter_42

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_74 = paddle._C_ops.transpose(add_76, [1, 0, 2])
        del add_76

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_74, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_74

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_77 = paddle._C_ops.add(layer_norm_48, dropout_54)
        del dropout_54, layer_norm_48

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_77, parameter_41, parameter_40, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_77, parameter_40, parameter_41

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_51, parameter_39, False, False)
        del parameter_39

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_78 = paddle._C_ops.add(matmul_68, parameter_38)
        del matmul_68, parameter_38

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_8 = paddle._C_ops.relu(add_78)
        del add_78

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_56, dropout_57 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_8, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_8

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_69 = paddle._C_ops.matmul(dropout_56, parameter_37, False, False)
        del dropout_56, parameter_37

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_79 = paddle._C_ops.add(matmul_69, parameter_36)
        del matmul_69, parameter_36

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_58, dropout_59 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_79, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_79

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_80 = paddle._C_ops.add(layer_norm_51, dropout_58)
        del dropout_58, layer_norm_51

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_80, parameter_35, parameter_34, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_80, parameter_34, parameter_35

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_54, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_81 = paddle._C_ops.add(layer_norm_54, stack_2)

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_75 = paddle._C_ops.transpose(add_81, [1, 0, 2])
        del add_81

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_76 = paddle._C_ops.transpose(layer_norm_54, [1, 0, 2])
        del layer_norm_54

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_70 = paddle._C_ops.matmul(transpose_75, parameter_33, False, False)
        del parameter_33

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_82 = paddle._C_ops.add(matmul_70, parameter_32)
        del matmul_70, parameter_32

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_42 = paddle._C_ops.reshape(add_82, full_int_array_7)
        del add_82

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_77 = paddle._C_ops.transpose(reshape_42, [0, 2, 1, 3])
        del reshape_42

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_71 = paddle._C_ops.matmul(transpose_75, parameter_31, False, False)
        del parameter_31, transpose_75

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_83 = paddle._C_ops.add(matmul_71, parameter_30)
        del matmul_71, parameter_30

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_72 = paddle._C_ops.matmul(transpose_76, parameter_29, False, False)
        del parameter_29

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_84 = paddle._C_ops.add(matmul_72, parameter_28)
        del matmul_72, parameter_28

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_43 = paddle._C_ops.reshape(add_83, full_int_array_7)
        del add_83

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_78 = paddle._C_ops.transpose(reshape_43, [0, 2, 1, 3])
        del reshape_43

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_44 = paddle._C_ops.reshape(add_84, full_int_array_7)
        del add_84

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_79 = paddle._C_ops.transpose(reshape_44, [0, 2, 1, 3])
        del reshape_44

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(transpose_77, full_13, float("0"), True)
        del transpose_77

        # pd_op.matmul: (1x8x100x100xf32) <- (1x8x100x32xf32, 1x8x100x32xf32)
        matmul_73 = paddle._C_ops.matmul(scale_17, transpose_78, False, True)
        del scale_17, transpose_78

        # pd_op.softmax: (1x8x100x100xf32) <- (1x8x100x100xf32)
        softmax_10 = paddle._C_ops.softmax(matmul_73, -1)
        del matmul_73

        # pd_op.dropout: (1x8x100x100xf32, 1x8x100x100xui8) <- (1x8x100x100xf32, None, 1xf32)
        dropout_60, dropout_61 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_10, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_10

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x100xf32, 1x8x100x32xf32)
        matmul_74 = paddle._C_ops.matmul(dropout_60, transpose_79, False, False)
        del dropout_60, transpose_79

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_80 = paddle._C_ops.transpose(matmul_74, [0, 2, 1, 3])
        del matmul_74

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_45 = paddle._C_ops.reshape(transpose_80, full_int_array_8)
        del transpose_80

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_75 = paddle._C_ops.matmul(reshape_45, parameter_27, False, False)
        del parameter_27, reshape_45

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_85 = paddle._C_ops.add(matmul_75, parameter_26)
        del matmul_75, parameter_26

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_81 = paddle._C_ops.transpose(add_85, [1, 0, 2])
        del add_85

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_82 = paddle._C_ops.transpose(transpose_76, [1, 0, 2])
        del transpose_76

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_62, dropout_63 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_81, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_81

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_86 = paddle._C_ops.add(transpose_82, dropout_62)
        del dropout_62, transpose_82

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_86, parameter_25, parameter_24, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_86, parameter_24, parameter_25

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_87 = paddle._C_ops.add(layer_norm_60, stack_2)
        del stack_2

        # pd_op.transpose: (1x100x256xf32) <- (100x1x256xf32)
        transpose_83 = paddle._C_ops.transpose(add_87, [1, 0, 2])
        del add_87

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_76 = paddle._C_ops.matmul(transpose_83, parameter_23, False, False)
        del parameter_23, transpose_83

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_88 = paddle._C_ops.add(matmul_76, parameter_22)
        del matmul_76, parameter_22

        # pd_op.reshape: (1x100x8x32xf32) <- (1x100x256xf32, 4xi64)
        reshape_46 = paddle._C_ops.reshape(add_88, full_int_array_7)
        del add_88

        # pd_op.transpose: (1x8x100x32xf32) <- (1x100x8x32xf32)
        transpose_84 = paddle._C_ops.transpose(reshape_46, [0, 2, 1, 3])
        del reshape_46

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_77 = paddle._C_ops.matmul(transpose_12, parameter_21, False, False)
        del parameter_21, transpose_12

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_89 = paddle._C_ops.add(matmul_77, parameter_20)
        del matmul_77, parameter_20

        # pd_op.matmul: (1x256x256xf32) <- (1x256x256xf32, 256x256xf32)
        matmul_78 = paddle._C_ops.matmul(transpose_13, parameter_19, False, False)
        del parameter_19, transpose_13

        # pd_op.add: (1x256x256xf32) <- (1x256x256xf32, 256xf32)
        add_90 = paddle._C_ops.add(matmul_78, parameter_18)
        del matmul_78, parameter_18

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_47 = paddle._C_ops.reshape(add_89, full_int_array_7)
        del add_89

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_85 = paddle._C_ops.transpose(reshape_47, [0, 2, 1, 3])
        del reshape_47

        # pd_op.reshape: (1x256x8x32xf32) <- (1x256x256xf32, 4xi64)
        reshape_48 = paddle._C_ops.reshape(add_90, full_int_array_7)
        del add_90, full_int_array_7

        # pd_op.transpose: (1x8x256x32xf32) <- (1x256x8x32xf32)
        transpose_86 = paddle._C_ops.transpose(reshape_48, [0, 2, 1, 3])
        del reshape_48

        # pd_op.scale: (1x8x100x32xf32) <- (1x8x100x32xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(transpose_84, full_13, float("0"), True)
        del full_13, transpose_84

        # pd_op.matmul: (1x8x100x256xf32) <- (1x8x100x32xf32, 1x8x256x32xf32)
        matmul_79 = paddle._C_ops.matmul(scale_18, transpose_85, False, True)
        del scale_18, transpose_85

        # pd_op.softmax: (1x8x100x256xf32) <- (1x8x100x256xf32)
        softmax_11 = paddle._C_ops.softmax(matmul_79, -1)
        del matmul_79

        # pd_op.dropout: (1x8x100x256xf32, 1x8x100x256xui8) <- (1x8x100x256xf32, None, 1xf32)
        dropout_64, dropout_65 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                softmax_11, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del softmax_11

        # pd_op.matmul: (1x8x100x32xf32) <- (1x8x100x256xf32, 1x8x256x32xf32)
        matmul_80 = paddle._C_ops.matmul(dropout_64, transpose_86, False, False)
        del dropout_64, transpose_86

        # pd_op.transpose: (1x100x8x32xf32) <- (1x8x100x32xf32)
        transpose_87 = paddle._C_ops.transpose(matmul_80, [0, 2, 1, 3])
        del matmul_80

        # pd_op.reshape: (1x100x256xf32) <- (1x100x8x32xf32, 3xi64)
        reshape_49 = paddle._C_ops.reshape(transpose_87, full_int_array_8)
        del full_int_array_8, transpose_87

        # pd_op.matmul: (1x100x256xf32) <- (1x100x256xf32, 256x256xf32)
        matmul_81 = paddle._C_ops.matmul(reshape_49, parameter_17, False, False)
        del parameter_17, reshape_49

        # pd_op.add: (1x100x256xf32) <- (1x100x256xf32, 256xf32)
        add_91 = paddle._C_ops.add(matmul_81, parameter_16)
        del matmul_81, parameter_16

        # pd_op.transpose: (100x1x256xf32) <- (1x100x256xf32)
        transpose_88 = paddle._C_ops.transpose(add_91, [1, 0, 2])
        del add_91

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_66, dropout_67 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                transpose_88, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del transpose_88

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_92 = paddle._C_ops.add(layer_norm_60, dropout_66)
        del dropout_66, layer_norm_60

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_92, parameter_15, parameter_14, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_92, parameter_14, parameter_15

        # pd_op.matmul: (100x1x2048xf32) <- (100x1x256xf32, 256x2048xf32)
        matmul_82 = paddle._C_ops.matmul(layer_norm_63, parameter_13, False, False)
        del parameter_13

        # pd_op.add: (100x1x2048xf32) <- (100x1x2048xf32, 2048xf32)
        add_93 = paddle._C_ops.add(matmul_82, parameter_12)
        del matmul_82, parameter_12

        # pd_op.relu: (100x1x2048xf32) <- (100x1x2048xf32)
        relu_9 = paddle._C_ops.relu(add_93)
        del add_93

        # pd_op.dropout: (100x1x2048xf32, 100x1x2048xui8) <- (100x1x2048xf32, None, 1xf32)
        dropout_68, dropout_69 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                relu_9, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del relu_9

        # pd_op.matmul: (100x1x256xf32) <- (100x1x2048xf32, 2048x256xf32)
        matmul_83 = paddle._C_ops.matmul(dropout_68, parameter_11, False, False)
        del dropout_68, parameter_11

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 256xf32)
        add_94 = paddle._C_ops.add(matmul_83, parameter_10)
        del matmul_83, parameter_10

        # pd_op.dropout: (100x1x256xf32, 100x1x256xui8) <- (100x1x256xf32, None, 1xf32)
        dropout_70, dropout_71 = (lambda x, f: f(x))(
            paddle._C_ops.dropout(
                add_94, None, full_14, True, "upscale_in_train", 0, False
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del add_94, full_14

        # pd_op.add: (100x1x256xf32) <- (100x1x256xf32, 100x1x256xf32)
        add_95 = paddle._C_ops.add(layer_norm_63, dropout_70)
        del dropout_70, layer_norm_63

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_95, parameter_9, parameter_8, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del add_95, parameter_8, parameter_9

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_66, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )

        # pd_op.layer_norm: (100x1x256xf32, 100x1xf32, 100x1xf32) <- (100x1x256xf32, 256xf32, 256xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                layer_norm_66, parameter_139, parameter_138, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del layer_norm_66, parameter_138, parameter_139

        # builtin.combine: ([100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32]) <- (100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32)
        combine_4 = [
            layer_norm_9,
            layer_norm_21,
            layer_norm_33,
            layer_norm_45,
            layer_norm_57,
            layer_norm_72,
        ]
        del (
            layer_norm_21,
            layer_norm_33,
            layer_norm_45,
            layer_norm_57,
            layer_norm_72,
            layer_norm_9,
        )

        # pd_op.stack: (6x100x1x256xf32) <- ([100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32, 100x1x256xf32])
        stack_3 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.transpose: (6x1x100x256xf32) <- (6x100x1x256xf32)
        transpose_89 = paddle._C_ops.transpose(stack_3, [0, 2, 1, 3])
        del stack_3

        # pd_op.transpose: (1x256x256xf32) <- (256x1x256xf32)
        transpose_90 = paddle._C_ops.transpose(transpose_1, [1, 2, 0])
        del transpose_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 256, 16, 16]

        # pd_op.reshape: (1x256x16x16xf32) <- (1x256x256xf32, 4xi64)
        reshape_50 = paddle._C_ops.reshape(transpose_90, full_int_array_9)
        del full_int_array_9, transpose_90

        # pd_op.matmul: (6x1x100x3xf32) <- (6x1x100x256xf32, 256x3xf32)
        matmul_84 = paddle._C_ops.matmul(transpose_89, parameter_7, False, False)
        del parameter_7

        # pd_op.add: (6x1x100x3xf32) <- (6x1x100x3xf32, 3xf32)
        add_96 = paddle._C_ops.add(matmul_84, parameter_6)
        del matmul_84, parameter_6

        # pd_op.slice: (1x100x3xf32) <- (6x1x100x3xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            add_96, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.matmul: (6x1x100x256xf32) <- (6x1x100x256xf32, 256x256xf32)
        matmul_85 = paddle._C_ops.matmul(transpose_89, parameter_5, False, False)
        del parameter_5, transpose_89

        # pd_op.add: (6x1x100x256xf32) <- (6x1x100x256xf32, 256xf32)
        add_97 = paddle._C_ops.add(matmul_85, parameter_4)
        del matmul_85, parameter_4

        # pd_op.relu: (6x1x100x256xf32) <- (6x1x100x256xf32)
        relu_10 = paddle._C_ops.relu(add_97)
        del add_97

        # pd_op.matmul: (6x1x100x256xf32) <- (6x1x100x256xf32, 256x256xf32)
        matmul_86 = paddle._C_ops.matmul(relu_10, parameter_3, False, False)
        del parameter_3, relu_10

        # pd_op.add: (6x1x100x256xf32) <- (6x1x100x256xf32, 256xf32)
        add_98 = paddle._C_ops.add(matmul_86, parameter_2)
        del matmul_86, parameter_2

        # pd_op.relu: (6x1x100x256xf32) <- (6x1x100x256xf32)
        relu_11 = paddle._C_ops.relu(add_98)
        del add_98

        # pd_op.matmul: (6x1x100x256xf32) <- (6x1x100x256xf32, 256x256xf32)
        matmul_87 = paddle._C_ops.matmul(relu_11, parameter_1, False, False)
        del parameter_1, relu_11

        # pd_op.add: (6x1x100x256xf32) <- (6x1x100x256xf32, 256xf32)
        add_99 = paddle._C_ops.add(matmul_87, parameter_0)
        del matmul_87, parameter_0

        # builtin.combine: ([6x1x100x256xf32, 1x256x128x128xf32]) <- (6x1x100x256xf32, 1x256x128x128xf32)
        combine_5 = [add_99, add_3]
        del add_3, add_99

        # pd_op.einsum: (6x1x100x128x128xf32, [0xf32, 0xf32], [6x1x100x256xf32, 1x256x128x128xf32]) <- ([6x1x100x256xf32, 1x256x128x128xf32])
        einsum_0, einsum_1, einsum_2 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_5, "lbqc,bchw->lbqhw"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_5

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_0,
            split_1,
        ) = einsum_1
        del einsum_1

        # builtin.split: (6x1x100x256xf32, 1x256x128x128xf32) <- ([6x1x100x256xf32, 1x256x128x128xf32])
        (
            split_2,
            split_3,
        ) = einsum_2
        del einsum_2

        # pd_op.slice: (1x100x128x128xf32) <- (6x1x100x128x128xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            einsum_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_2

        # pd_op.slice: (5x1x100x3xf32) <- (6x1x100x3xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            add_96, [0], full_int_array_4, full_int_array_1, [1], []
        )
        del add_96

        # pd_op.slice: (5x1x100x128x128xf32) <- (6x1x100x128x128xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            einsum_0, [0], full_int_array_4, full_int_array_1, [1], []
        )
        del einsum_0

        # pd_op.slice: (1x100x3xf32) <- (5x1x100x3xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_4, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (1x100x128x128xf32) <- (5x1x100x128x128xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_4, full_int_array_6, [1], [0]
        )

        # pd_op.slice: (1x100x3xf32) <- (5x1x100x3xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_6, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (1x100x128x128xf32) <- (5x1x100x128x128xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_6, full_int_array_5, [1], [0]
        )

        # pd_op.slice: (1x100x3xf32) <- (5x1x100x3xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_5, full_int_array_3, [1], [0]
        )

        # pd_op.slice: (1x100x128x128xf32) <- (5x1x100x128x128xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_5, full_int_array_3, [1], [0]
        )
        del full_int_array_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [4]

        # pd_op.slice: (1x100x3xf32) <- (5x1x100x3xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_3, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (1x100x128x128xf32) <- (5x1x100x128x128xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_3, full_int_array_10, [1], [0]
        )
        del full_int_array_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [5]

        # pd_op.slice: (1x100x3xf32) <- (5x1x100x3xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_10, full_int_array_11, [1], [0]
        )
        del slice_4

        # pd_op.slice: (1x100x128x128xf32) <- (5x1x100x128x128xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_10, full_int_array_11, [1], [0]
        )
        del full_int_array_10, full_int_array_11, slice_5

        # pd_op.bilinear_interp: (1x100x512x512xf32) <- (1x100x128x128xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(
            slice_3, None, None, None, "NCHW", -1, 512, 512, [], "bilinear", False, 0
        )
        del slice_3

        # pd_op.slice: (100x3xf32) <- (1x100x3xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            slice_2, [0], full_int_array_4, full_int_array_6, [1], [0]
        )
        del slice_2

        # pd_op.slice: (100x512x512xf32) <- (1x100x512x512xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            bilinear_interp_0, [0], full_int_array_4, full_int_array_6, [1], [0]
        )
        del bilinear_interp_0

        # pd_op.softmax: (100x3xf32) <- (100x3xf32)
        softmax_12 = paddle._C_ops.softmax(slice_16, -1)
        del slice_16

        # pd_op.slice: (100x2xf32) <- (100x3xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            softmax_12, [1], full_int_array_4, full_int_array_1, [1], []
        )
        del full_int_array_1, softmax_12

        # pd_op.sigmoid: (100x512x512xf32) <- (100x512x512xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(slice_17)
        del slice_17

        # builtin.combine: ([100x2xf32, 100x512x512xf32]) <- (100x2xf32, 100x512x512xf32)
        combine_6 = [slice_18, sigmoid_0]
        del sigmoid_0, slice_18

        # pd_op.einsum: (2x512x512xf32, [0xf32, 0xf32], [100x2xf32, 100x512x512xf32]) <- ([100x2xf32, 100x512x512xf32])
        einsum_3, einsum_4, einsum_5 = (lambda x, f: f(x))(
            paddle._C_ops.einsum(combine_6, "qc,qhw->chw"),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del combine_6

        # builtin.split: (0xf32, 0xf32) <- ([0xf32, 0xf32])
        (
            split_4,
            split_5,
        ) = einsum_4
        del einsum_4

        # builtin.split: (100x2xf32, 100x512x512xf32) <- ([100x2xf32, 100x512x512xf32])
        (
            split_6,
            split_7,
        ) = einsum_5
        del einsum_5

        # pd_op.unsqueeze: (1x2x512x512xf32) <- (2x512x512xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(einsum_3, full_int_array_4)
        del einsum_3

        # pd_op.bilinear_interp: (1x2x512x512xf32) <- (1x2x512x512xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(
            unsqueeze_3,
            None,
            None,
            None,
            "NCHW",
            -1,
            512,
            512,
            [],
            "bilinear",
            False,
            0,
        )
        del unsqueeze_3

        # pd_op.slice: (2x512x512xf32) <- (1x2x512x512xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            bilinear_interp_1, [0], full_int_array_4, full_int_array_6, [1], [0]
        )
        del bilinear_interp_1, full_int_array_6

        # pd_op.unsqueeze: (1x2x512x512xf32) <- (2x512x512xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_19, full_int_array_4)
        del full_int_array_4, slice_19

        return unsqueeze_0
