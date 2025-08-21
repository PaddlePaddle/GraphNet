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
        parameter_215,
        parameter_216,
        parameter_217,
        parameter_218,
        data_0,
    ):
        # pd_op.conv2d: (64x192x56x56xf32) <- (64x3x224x224xf32, 192x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_218, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x56x56xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_217,
                parameter_216,
                parameter_215,
                parameter_214,
                False,
                float("0.1"),
                float("1e-05"),
                "NCHW",
                False,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [48, 144]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_54 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_53 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_52 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_51 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_50 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_49 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.split: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x192x56x56xf32, 2xi64, 1xi32)
        split_56 = paddle._C_ops.split(batch_norm__0, full_int_array_1, full_0)

        # builtin.split: (64x48x56x56xf32, 64x144x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32])
        (
            split_0,
            split_1,
        ) = split_56

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_0, parameter_213, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x48x56x56xf32, 64x144x56x56xf32)
        combine_0 = [conv2d_1, split_1]

        # pd_op.concat: (64x192x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (64x384x56x56xf32) <- (64x192x56x56xf32, 384x192x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__164,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
                parameter_211,
                parameter_210,
                parameter_209,
                parameter_208,
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

        # pd_op.relu: (64x384x56x56xf32) <- (64x384x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__164)

        # pd_op.conv2d: (64x192x56x56xf32) <- (64x384x56x56xf32, 192x384x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (64x192x56x56xf32) <- (64x192x56x56xf32, 64x192x56x56xf32)
        add_0 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.split: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x192x56x56xf32, 2xi64, 1xi32)
        split_57 = paddle._C_ops.split(add_0, full_int_array_1, full_0)

        # builtin.split: (64x48x56x56xf32, 64x144x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32])
        (
            split_2,
            split_3,
        ) = split_57

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            split_2, parameter_206, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x48x56x56xf32, 64x144x56x56xf32)
        combine_1 = [conv2d_4, split_3]

        # pd_op.concat: (64x192x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.conv2d: (64x384x56x56xf32) <- (64x192x56x56xf32, 384x192x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_1, parameter_205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__165,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_5,
                parameter_204,
                parameter_203,
                parameter_202,
                parameter_201,
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

        # pd_op.relu: (64x384x56x56xf32) <- (64x384x56x56xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__165)

        # pd_op.conv2d: (64x192x56x56xf32) <- (64x384x56x56xf32, 192x384x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_1, parameter_200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [],
            float("0.988889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [64, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_28 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_29 = paddle._C_ops.add(full_1, uniform_0)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_29)

        # pd_op.divide: (64x192x56x56xf32) <- (64x192x56x56xf32, xf32)
        divide_0 = paddle._C_ops.divide(conv2d_6, full_1)

        # pd_op.multiply: (64x192x56x56xf32) <- (64x192x56x56xf32, 64x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (64x192x56x56xf32) <- (64x192x56x56xf32, 64x192x56x56xf32)
        add_1 = paddle._C_ops.add(add_0, multiply_0)

        # pd_op.split: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x192x56x56xf32, 2xi64, 1xi32)
        split_58 = paddle._C_ops.split(add_1, full_int_array_1, full_0)

        # builtin.split: (64x48x56x56xf32, 64x144x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32])
        (
            split_4,
            split_5,
        ) = split_58

        # pd_op.conv2d: (64x48x56x56xf32) <- (64x48x56x56xf32, 48x48x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            split_4, parameter_199, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x48x56x56xf32, 64x144x56x56xf32]) <- (64x48x56x56xf32, 64x144x56x56xf32)
        combine_2 = [conv2d_7, split_5]

        # pd_op.concat: (64x192x56x56xf32) <- ([64x48x56x56xf32, 64x144x56x56xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.conv2d: (64x384x56x56xf32) <- (64x192x56x56xf32, 384x192x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_2, parameter_198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x56x56xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__166,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_8,
                parameter_197,
                parameter_196,
                parameter_195,
                parameter_194,
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

        # pd_op.relu: (64x384x56x56xf32) <- (64x384x56x56xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__166)

        # pd_op.conv2d: (64x192x56x56xf32) <- (64x384x56x56xf32, 192x384x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_193, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [],
            float("0.977778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_30 = paddle._C_ops.add(full_2, uniform_1)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_30)

        # pd_op.divide: (64x192x56x56xf32) <- (64x192x56x56xf32, xf32)
        divide_1 = paddle._C_ops.divide(conv2d_9, full_2)

        # pd_op.multiply: (64x192x56x56xf32) <- (64x192x56x56xf32, 64x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (64x192x56x56xf32) <- (64x192x56x56xf32, 64x192x56x56xf32)
        add_2 = paddle._C_ops.add(add_1, multiply_1)

        # pd_op.conv2d: (64x384x28x28xf32) <- (64x192x56x56xf32, 384x192x2x2xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            add_2, parameter_192, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (64x384x28x28xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_191,
                parameter_190,
                parameter_189,
                parameter_188,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [96, 288]

        # pd_op.split: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x384x28x28xf32, 2xi64, 1xi32)
        split_59 = paddle._C_ops.split(batch_norm__21, full_int_array_3, full_0)

        # builtin.split: (64x96x28x28xf32, 64x288x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32])
        (
            split_6,
            split_7,
        ) = split_59

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            split_6, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x96x28x28xf32, 64x288x28x28xf32)
        combine_3 = [conv2d_11, split_7]

        # pd_op.concat: (64x384x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.conv2d: (64x768x28x28xf32) <- (64x384x28x28xf32, 768x384x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            concat_3, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__167,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_185,
                parameter_184,
                parameter_183,
                parameter_182,
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

        # pd_op.relu: (64x768x28x28xf32) <- (64x768x28x28xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__167)

        # pd_op.conv2d: (64x384x28x28xf32) <- (64x768x28x28xf32, 384x768x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_3, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [],
            float("0.966667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_31 = paddle._C_ops.add(full_3, uniform_2)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_31)

        # pd_op.divide: (64x384x28x28xf32) <- (64x384x28x28xf32, xf32)
        divide_2 = paddle._C_ops.divide(conv2d_13, full_3)

        # pd_op.multiply: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x384x28x28xf32)
        add_3 = paddle._C_ops.add(batch_norm__21, multiply_2)

        # pd_op.split: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x384x28x28xf32, 2xi64, 1xi32)
        split_60 = paddle._C_ops.split(add_3, full_int_array_3, full_0)

        # builtin.split: (64x96x28x28xf32, 64x288x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32])
        (
            split_8,
            split_9,
        ) = split_60

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            split_8, parameter_180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x96x28x28xf32, 64x288x28x28xf32)
        combine_4 = [conv2d_14, split_9]

        # pd_op.concat: (64x384x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)

        # pd_op.conv2d: (64x768x28x28xf32) <- (64x384x28x28xf32, 768x384x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_4, parameter_179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__168,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_178,
                parameter_177,
                parameter_176,
                parameter_175,
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

        # pd_op.relu: (64x768x28x28xf32) <- (64x768x28x28xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.conv2d: (64x384x28x28xf32) <- (64x768x28x28xf32, 384x768x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_4, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("0.955556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_32 = paddle._C_ops.add(full_4, uniform_3)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_32)

        # pd_op.divide: (64x384x28x28xf32) <- (64x384x28x28xf32, xf32)
        divide_3 = paddle._C_ops.divide(conv2d_16, full_4)

        # pd_op.multiply: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x384x28x28xf32)
        add_4 = paddle._C_ops.add(add_3, multiply_3)

        # pd_op.split: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x384x28x28xf32, 2xi64, 1xi32)
        split_61 = paddle._C_ops.split(add_4, full_int_array_3, full_0)

        # builtin.split: (64x96x28x28xf32, 64x288x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32])
        (
            split_10,
            split_11,
        ) = split_61

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            split_10, parameter_173, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x96x28x28xf32, 64x288x28x28xf32)
        combine_5 = [conv2d_17, split_11]

        # pd_op.concat: (64x384x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)

        # pd_op.conv2d: (64x768x28x28xf32) <- (64x384x28x28xf32, 768x384x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            concat_5, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__169,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_171,
                parameter_170,
                parameter_169,
                parameter_168,
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

        # pd_op.relu: (64x768x28x28xf32) <- (64x768x28x28xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__169)

        # pd_op.conv2d: (64x384x28x28xf32) <- (64x768x28x28xf32, 384x768x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_5, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.944444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_33 = paddle._C_ops.add(full_5, uniform_4)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_33)

        # pd_op.divide: (64x384x28x28xf32) <- (64x384x28x28xf32, xf32)
        divide_4 = paddle._C_ops.divide(conv2d_19, full_5)

        # pd_op.multiply: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x384x28x28xf32)
        add_5 = paddle._C_ops.add(add_4, multiply_4)

        # pd_op.split: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x384x28x28xf32, 2xi64, 1xi32)
        split_62 = paddle._C_ops.split(add_5, full_int_array_3, full_0)

        # builtin.split: (64x96x28x28xf32, 64x288x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32])
        (
            split_12,
            split_13,
        ) = split_62

        # pd_op.conv2d: (64x96x28x28xf32) <- (64x96x28x28xf32, 96x96x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            split_12, parameter_166, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x96x28x28xf32, 64x288x28x28xf32]) <- (64x96x28x28xf32, 64x288x28x28xf32)
        combine_6 = [conv2d_20, split_13]

        # pd_op.concat: (64x384x28x28xf32) <- ([64x96x28x28xf32, 64x288x28x28xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.conv2d: (64x768x28x28xf32) <- (64x384x28x28xf32, 768x384x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            concat_6, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (64x768x28x28xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__170,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_164,
                parameter_163,
                parameter_162,
                parameter_161,
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

        # pd_op.relu: (64x768x28x28xf32) <- (64x768x28x28xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__170)

        # pd_op.conv2d: (64x384x28x28xf32) <- (64x768x28x28xf32, 384x768x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_6, parameter_160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.933333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_34 = paddle._C_ops.add(full_6, uniform_5)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_34)

        # pd_op.divide: (64x384x28x28xf32) <- (64x384x28x28xf32, xf32)
        divide_5 = paddle._C_ops.divide(conv2d_22, full_6)

        # pd_op.multiply: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (64x384x28x28xf32) <- (64x384x28x28xf32, 64x384x28x28xf32)
        add_6 = paddle._C_ops.add(add_5, multiply_5)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x384x28x28xf32, 768x384x2x2xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_6, parameter_159, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (64x768x14x14xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [192, 576]

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_63 = paddle._C_ops.split(batch_norm__47, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_14,
            split_15,
        ) = split_63

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_14, parameter_154, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_7 = [conv2d_24, split_15]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__171,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__171)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.922222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_35 = paddle._C_ops.add(full_7, uniform_6)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_35)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(conv2d_26, full_7)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_7 = paddle._C_ops.add(batch_norm__47, multiply_6)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_64 = paddle._C_ops.split(add_7, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_64

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_16, parameter_147, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_8 = [conv2d_27, split_17]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__172,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_145,
                parameter_144,
                parameter_143,
                parameter_142,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__172)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.911111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_36 = paddle._C_ops.add(full_8, uniform_7)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_36)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(conv2d_29, full_8)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, multiply_7)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_65 = paddle._C_ops.split(add_8, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_18,
            split_19,
        ) = split_65

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_18, parameter_140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_9 = [conv2d_30, split_19]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__173,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__173)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [], float("0.9"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_37 = paddle._C_ops.add(full_9, uniform_8)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_37)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(conv2d_32, full_9)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, multiply_8)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_66 = paddle._C_ops.split(add_9, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_20,
            split_21,
        ) = split_66

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_20, parameter_133, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_10 = [conv2d_33, split_21]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__174,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_131,
                parameter_130,
                parameter_129,
                parameter_128,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__174)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.888889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_38 = paddle._C_ops.add(full_10, uniform_9)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_38)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(conv2d_35, full_10)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, multiply_9)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_67 = paddle._C_ops.split(add_10, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_67

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            split_22, parameter_126, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_11 = [conv2d_36, split_23]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            concat_11, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__175,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
                parameter_124,
                parameter_123,
                parameter_122,
                parameter_121,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__175)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_11, parameter_120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.877778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_39 = paddle._C_ops.add(full_11, uniform_10)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_39)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_10 = paddle._C_ops.divide(conv2d_38, full_11)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, multiply_10)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_68 = paddle._C_ops.split(add_11, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_24,
            split_25,
        ) = split_68

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            split_24, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_12 = [conv2d_39, split_25]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_12, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__176,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
                parameter_117,
                parameter_116,
                parameter_115,
                parameter_114,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__176)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_12, parameter_113, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.866667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_40 = paddle._C_ops.add(full_12, uniform_11)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_40)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_11 = paddle._C_ops.divide(conv2d_41, full_12)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_12 = paddle._C_ops.add(add_11, multiply_11)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_69 = paddle._C_ops.split(add_12, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_26,
            split_27,
        ) = split_69

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            split_26, parameter_112, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_13 = [conv2d_42, split_27]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            concat_13, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__177,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_43,
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__177)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_13, parameter_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.855556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_41 = paddle._C_ops.add(full_13, uniform_12)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_41)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_12 = paddle._C_ops.divide(conv2d_44, full_13)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_13 = paddle._C_ops.add(add_12, multiply_12)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_70 = paddle._C_ops.split(add_13, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_70

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            split_28, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_14 = [conv2d_45, split_29]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            concat_14, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__178,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__178)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_14, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.844444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_42 = paddle._C_ops.add(full_14, uniform_13)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_42)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_13 = paddle._C_ops.divide(conv2d_47, full_14)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_14 = paddle._C_ops.add(add_13, multiply_13)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_71 = paddle._C_ops.split(add_14, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_30,
            split_31,
        ) = split_71

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            split_30, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_15 = [conv2d_48, split_31]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            concat_15, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__179,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_49,
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__179)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_15, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.833333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_43 = paddle._C_ops.add(full_15, uniform_14)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_43)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_14 = paddle._C_ops.divide(conv2d_50, full_15)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_15 = paddle._C_ops.add(add_14, multiply_14)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_72 = paddle._C_ops.split(add_15, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_32,
            split_33,
        ) = split_72

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            split_32, parameter_91, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_16 = [conv2d_51, split_33]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            concat_16, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__180,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_52,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            relu_16, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [],
            float("0.822222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_44 = paddle._C_ops.add(full_16, uniform_15)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_44)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_15 = paddle._C_ops.divide(conv2d_53, full_16)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_16 = paddle._C_ops.add(add_15, multiply_15)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_73 = paddle._C_ops.split(add_16, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_34,
            split_35,
        ) = split_73

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            split_34, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_17 = [conv2d_54, split_35]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            concat_17, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__181,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_55,
                parameter_82,
                parameter_81,
                parameter_80,
                parameter_79,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__181)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            relu_17, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [],
            float("0.811111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_45 = paddle._C_ops.add(full_17, uniform_16)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_45)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_16 = paddle._C_ops.divide(conv2d_56, full_17)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_17 = paddle._C_ops.add(add_16, multiply_16)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_74 = paddle._C_ops.split(add_17, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_36,
            split_37,
        ) = split_74

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            split_36, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_18 = [conv2d_57, split_37]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            concat_18, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__182,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_58,
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__182)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            relu_18, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_18 = paddle._C_ops.full(
            [], float("0.8"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_17 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_46 = paddle._C_ops.add(full_18, uniform_17)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_17 = paddle._C_ops.floor(add_46)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_17 = paddle._C_ops.divide(conv2d_59, full_18)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_17, floor_17)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_18 = paddle._C_ops.add(add_17, multiply_17)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_75 = paddle._C_ops.split(add_18, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_38,
            split_39,
        ) = split_75

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            split_38, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_19 = [conv2d_60, split_39]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            concat_19, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__183,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_61,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__183)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            relu_19, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_19 = paddle._C_ops.full(
            [],
            float("0.788889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_18 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_47 = paddle._C_ops.add(full_19, uniform_18)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_18 = paddle._C_ops.floor(add_47)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_18 = paddle._C_ops.divide(conv2d_62, full_19)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_18, floor_18)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_19 = paddle._C_ops.add(add_18, multiply_18)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_76 = paddle._C_ops.split(add_19, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_40,
            split_41,
        ) = split_76

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            split_40, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_20 = [conv2d_63, split_41]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            concat_20, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__184,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_64,
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__184)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            relu_20, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_20 = paddle._C_ops.full(
            [],
            float("0.777778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_19 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_48 = paddle._C_ops.add(full_20, uniform_19)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_19 = paddle._C_ops.floor(add_48)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_19 = paddle._C_ops.divide(conv2d_65, full_20)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_19, floor_19)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_20 = paddle._C_ops.add(add_19, multiply_19)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_77 = paddle._C_ops.split(add_20, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_42,
            split_43,
        ) = split_77

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            split_42, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_21 = [conv2d_66, split_43]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            concat_21, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__185,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_67,
                parameter_54,
                parameter_53,
                parameter_52,
                parameter_51,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__185)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            relu_21, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_21 = paddle._C_ops.full(
            [],
            float("0.766667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_20 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_49 = paddle._C_ops.add(full_21, uniform_20)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_20 = paddle._C_ops.floor(add_49)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_20 = paddle._C_ops.divide(conv2d_68, full_21)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_20, floor_20)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_21 = paddle._C_ops.add(add_20, multiply_20)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_78 = paddle._C_ops.split(add_21, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_44,
            split_45,
        ) = split_78

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            split_44, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_22 = [conv2d_69, split_45]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            concat_22, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__186,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_70,
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__186)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(
            relu_22, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_22 = paddle._C_ops.full(
            [],
            float("0.755556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_21 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_50 = paddle._C_ops.add(full_22, uniform_21)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_21 = paddle._C_ops.floor(add_50)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_21 = paddle._C_ops.divide(conv2d_71, full_22)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_21 = paddle._C_ops.multiply(divide_21, floor_21)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_22 = paddle._C_ops.add(add_21, multiply_21)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_79 = paddle._C_ops.split(add_22, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_46,
            split_47,
        ) = split_79

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(
            split_46, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_23 = [conv2d_72, split_47]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(
            concat_23, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__187,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_73,
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__187)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(
            relu_23, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_23 = paddle._C_ops.full(
            [],
            float("0.744444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_22 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_51 = paddle._C_ops.add(full_23, uniform_22)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_22 = paddle._C_ops.floor(add_51)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_22 = paddle._C_ops.divide(conv2d_74, full_23)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_22 = paddle._C_ops.multiply(divide_22, floor_22)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_23 = paddle._C_ops.add(add_22, multiply_22)

        # pd_op.split: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x768x14x14xf32, 2xi64, 1xi32)
        split_80 = paddle._C_ops.split(add_23, full_int_array_4, full_0)

        # builtin.split: (64x192x14x14xf32, 64x576x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32])
        (
            split_48,
            split_49,
        ) = split_80

        # pd_op.conv2d: (64x192x14x14xf32) <- (64x192x14x14xf32, 192x192x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(
            split_48, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x192x14x14xf32, 64x576x14x14xf32]) <- (64x192x14x14xf32, 64x576x14x14xf32)
        combine_24 = [conv2d_75, split_49]

        # pd_op.concat: (64x768x14x14xf32) <- ([64x192x14x14xf32, 64x576x14x14xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_0)

        # pd_op.conv2d: (64x1536x14x14xf32) <- (64x768x14x14xf32, 1536x768x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(
            concat_24, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x14x14xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__188,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_76,
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

        # pd_op.relu: (64x1536x14x14xf32) <- (64x1536x14x14xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__188)

        # pd_op.conv2d: (64x768x14x14xf32) <- (64x1536x14x14xf32, 768x1536x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(
            relu_24, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_24 = paddle._C_ops.full(
            [],
            float("0.733333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_23 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_52 = paddle._C_ops.add(full_24, uniform_23)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_23 = paddle._C_ops.floor(add_52)

        # pd_op.divide: (64x768x14x14xf32) <- (64x768x14x14xf32, xf32)
        divide_23 = paddle._C_ops.divide(conv2d_77, full_24)

        # pd_op.multiply: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x1x1x1xf32)
        multiply_23 = paddle._C_ops.multiply(divide_23, floor_23)

        # pd_op.add: (64x768x14x14xf32) <- (64x768x14x14xf32, 64x768x14x14xf32)
        add_24 = paddle._C_ops.add(add_23, multiply_23)

        # pd_op.conv2d: (64x1536x7x7xf32) <- (64x768x14x14xf32, 1536x768x2x2xf32)
        conv2d_78 = paddle._C_ops.conv2d(
            add_24, parameter_28, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32, -1xui8) <- (64x1536x7x7xf32, 1536xf32, 1536xf32, 1536xf32, 1536xf32)
        (
            batch_norm__143,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_78,
                parameter_27,
                parameter_26,
                parameter_25,
                parameter_24,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [384, 1152]

        # pd_op.split: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x1536x7x7xf32, 2xi64, 1xi32)
        split_81 = paddle._C_ops.split(batch_norm__143, full_int_array_5, full_0)

        # builtin.split: (64x384x7x7xf32, 64x1152x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32])
        (
            split_50,
            split_51,
        ) = split_81

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(
            split_50, parameter_23, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x384x7x7xf32, 64x1152x7x7xf32)
        combine_25 = [conv2d_79, split_51]

        # pd_op.concat: (64x1536x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_0)

        # pd_op.conv2d: (64x3072x7x7xf32) <- (64x1536x7x7xf32, 3072x1536x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(
            concat_25, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32, -1xui8) <- (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32)
        (
            batch_norm__189,
            batch_norm__149,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_80,
                parameter_21,
                parameter_20,
                parameter_19,
                parameter_18,
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

        # pd_op.relu: (64x3072x7x7xf32) <- (64x3072x7x7xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__189)

        # pd_op.conv2d: (64x1536x7x7xf32) <- (64x3072x7x7xf32, 1536x3072x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(
            relu_25, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_25 = paddle._C_ops.full(
            [],
            float("0.722222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_24 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_53 = paddle._C_ops.add(full_25, uniform_24)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_24 = paddle._C_ops.floor(add_53)

        # pd_op.divide: (64x1536x7x7xf32) <- (64x1536x7x7xf32, xf32)
        divide_24 = paddle._C_ops.divide(conv2d_81, full_25)

        # pd_op.multiply: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1x1x1xf32)
        multiply_24 = paddle._C_ops.multiply(divide_24, floor_24)

        # pd_op.add: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1536x7x7xf32)
        add_25 = paddle._C_ops.add(batch_norm__143, multiply_24)

        # pd_op.split: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x1536x7x7xf32, 2xi64, 1xi32)
        split_82 = paddle._C_ops.split(add_25, full_int_array_5, full_0)

        # builtin.split: (64x384x7x7xf32, 64x1152x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32])
        (
            split_52,
            split_53,
        ) = split_82

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(
            split_52, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x384x7x7xf32, 64x1152x7x7xf32)
        combine_26 = [conv2d_82, split_53]

        # pd_op.concat: (64x1536x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_0)

        # pd_op.conv2d: (64x3072x7x7xf32) <- (64x1536x7x7xf32, 3072x1536x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(
            concat_26, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32, -1xui8) <- (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32)
        (
            batch_norm__190,
            batch_norm__154,
            batch_norm__155,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_83,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
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

        # pd_op.relu: (64x3072x7x7xf32) <- (64x3072x7x7xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__190)

        # pd_op.conv2d: (64x1536x7x7xf32) <- (64x3072x7x7xf32, 1536x3072x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(
            relu_26, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_26 = paddle._C_ops.full(
            [],
            float("0.711111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_25 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_54 = paddle._C_ops.add(full_26, uniform_25)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_25 = paddle._C_ops.floor(add_54)

        # pd_op.divide: (64x1536x7x7xf32) <- (64x1536x7x7xf32, xf32)
        divide_25 = paddle._C_ops.divide(conv2d_84, full_26)

        # pd_op.multiply: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1x1x1xf32)
        multiply_25 = paddle._C_ops.multiply(divide_25, floor_25)

        # pd_op.add: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1536x7x7xf32)
        add_26 = paddle._C_ops.add(add_25, multiply_25)

        # pd_op.split: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x1536x7x7xf32, 2xi64, 1xi32)
        split_83 = paddle._C_ops.split(add_26, full_int_array_5, full_0)

        # builtin.split: (64x384x7x7xf32, 64x1152x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32])
        (
            split_54,
            split_55,
        ) = split_83

        # pd_op.conv2d: (64x384x7x7xf32) <- (64x384x7x7xf32, 384x384x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(
            split_54, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([64x384x7x7xf32, 64x1152x7x7xf32]) <- (64x384x7x7xf32, 64x1152x7x7xf32)
        combine_27 = [conv2d_85, split_55]

        # pd_op.concat: (64x1536x7x7xf32) <- ([64x384x7x7xf32, 64x1152x7x7xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_0)

        # pd_op.conv2d: (64x3072x7x7xf32) <- (64x1536x7x7xf32, 3072x1536x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(
            concat_27, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32, -1xui8) <- (64x3072x7x7xf32, 3072xf32, 3072xf32, 3072xf32, 3072xf32)
        (
            batch_norm__191,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_86,
                parameter_7,
                parameter_6,
                parameter_5,
                parameter_4,
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

        # pd_op.relu: (64x3072x7x7xf32) <- (64x3072x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__191)

        # pd_op.conv2d: (64x1536x7x7xf32) <- (64x3072x7x7xf32, 1536x3072x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(
            relu_27, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_27 = paddle._C_ops.full(
            [], float("0.7"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.uniform: (64x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_26 = paddle._C_ops.uniform(
            full_int_array_2,
            paddle.float32,
            full_28,
            full_29,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (64x1x1x1xf32) <- (xf32, 64x1x1x1xf32)
        add_55 = paddle._C_ops.add(full_27, uniform_26)

        # pd_op.floor: (64x1x1x1xf32) <- (64x1x1x1xf32)
        floor_26 = paddle._C_ops.floor(add_55)

        # pd_op.divide: (64x1536x7x7xf32) <- (64x1536x7x7xf32, xf32)
        divide_26 = paddle._C_ops.divide(conv2d_87, full_27)

        # pd_op.multiply: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1x1x1xf32)
        multiply_26 = paddle._C_ops.multiply(divide_26, floor_26)

        # pd_op.add: (64x1536x7x7xf32) <- (64x1536x7x7xf32, 64x1536x7x7xf32)
        add_27 = paddle._C_ops.add(add_26, multiply_26)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (64x1536x1x1xf32) <- (64x1536x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_27,
            full_int_array_0,
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

        # pd_op.conv2d: (64x1280x1x1xf32) <- (64x1536x1x1xf32, 1280x1536x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.relu: (64x1280x1x1xf32) <- (64x1280x1x1xf32)
        relu_28 = paddle._C_ops.relu(conv2d_88)

        # pd_op.flatten: (64x1280xf32) <- (64x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_28, 1, 3)

        # pd_op.matmul: (64x102xf32) <- (64x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)

        # pd_op.add: (64x102xf32) <- (64x102xf32, 102xf32)
        add_28 = paddle._C_ops.add(matmul_0, parameter_0)
        return (
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            full_0,
            split_0,
            split_1,
            conv2d_1,
            assign_0,
            concat_0,
            conv2d_2,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            relu_0,
            conv2d_3,
            add_0,
            assign_1,
            split_2,
            split_3,
            conv2d_4,
            assign_2,
            concat_1,
            conv2d_5,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            relu_1,
            conv2d_6,
            full_1,
            floor_0,
            divide_0,
            multiply_0,
            add_1,
            assign_3,
            split_4,
            split_5,
            conv2d_7,
            assign_4,
            concat_2,
            conv2d_8,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            relu_2,
            conv2d_9,
            full_2,
            floor_1,
            divide_1,
            multiply_1,
            add_2,
            conv2d_10,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            assign_5,
            split_6,
            split_7,
            conv2d_11,
            assign_6,
            concat_3,
            conv2d_12,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            relu_3,
            conv2d_13,
            full_3,
            floor_2,
            divide_2,
            multiply_2,
            add_3,
            assign_7,
            split_8,
            split_9,
            conv2d_14,
            assign_8,
            concat_4,
            conv2d_15,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            relu_4,
            conv2d_16,
            full_4,
            floor_3,
            divide_3,
            multiply_3,
            add_4,
            assign_9,
            split_10,
            split_11,
            conv2d_17,
            assign_10,
            concat_5,
            conv2d_18,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            relu_5,
            conv2d_19,
            full_5,
            floor_4,
            divide_4,
            multiply_4,
            add_5,
            assign_11,
            split_12,
            split_13,
            conv2d_20,
            assign_12,
            concat_6,
            conv2d_21,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            relu_6,
            conv2d_22,
            full_6,
            floor_5,
            divide_5,
            multiply_5,
            add_6,
            conv2d_23,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            assign_13,
            split_14,
            split_15,
            conv2d_24,
            assign_14,
            concat_7,
            conv2d_25,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            relu_7,
            conv2d_26,
            full_7,
            floor_6,
            divide_6,
            multiply_6,
            add_7,
            assign_15,
            split_16,
            split_17,
            conv2d_27,
            assign_16,
            concat_8,
            conv2d_28,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            relu_8,
            conv2d_29,
            full_8,
            floor_7,
            divide_7,
            multiply_7,
            add_8,
            assign_17,
            split_18,
            split_19,
            conv2d_30,
            assign_18,
            concat_9,
            conv2d_31,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            relu_9,
            conv2d_32,
            full_9,
            floor_8,
            divide_8,
            multiply_8,
            add_9,
            assign_19,
            split_20,
            split_21,
            conv2d_33,
            assign_20,
            concat_10,
            conv2d_34,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            relu_10,
            conv2d_35,
            full_10,
            floor_9,
            divide_9,
            multiply_9,
            add_10,
            assign_21,
            split_22,
            split_23,
            conv2d_36,
            assign_22,
            concat_11,
            conv2d_37,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            relu_11,
            conv2d_38,
            full_11,
            floor_10,
            divide_10,
            multiply_10,
            add_11,
            assign_23,
            split_24,
            split_25,
            conv2d_39,
            assign_24,
            concat_12,
            conv2d_40,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            relu_12,
            conv2d_41,
            full_12,
            floor_11,
            divide_11,
            multiply_11,
            add_12,
            assign_25,
            split_26,
            split_27,
            conv2d_42,
            assign_26,
            concat_13,
            conv2d_43,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            relu_13,
            conv2d_44,
            full_13,
            floor_12,
            divide_12,
            multiply_12,
            add_13,
            assign_27,
            split_28,
            split_29,
            conv2d_45,
            assign_28,
            concat_14,
            conv2d_46,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            relu_14,
            conv2d_47,
            full_14,
            floor_13,
            divide_13,
            multiply_13,
            add_14,
            assign_29,
            split_30,
            split_31,
            conv2d_48,
            assign_30,
            concat_15,
            conv2d_49,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
            relu_15,
            conv2d_50,
            full_15,
            floor_14,
            divide_14,
            multiply_14,
            add_15,
            assign_31,
            split_32,
            split_33,
            conv2d_51,
            assign_32,
            concat_16,
            conv2d_52,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            relu_16,
            conv2d_53,
            full_16,
            floor_15,
            divide_15,
            multiply_15,
            add_16,
            assign_33,
            split_34,
            split_35,
            conv2d_54,
            assign_34,
            concat_17,
            conv2d_55,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            relu_17,
            conv2d_56,
            full_17,
            floor_16,
            divide_16,
            multiply_16,
            add_17,
            assign_35,
            split_36,
            split_37,
            conv2d_57,
            assign_36,
            concat_18,
            conv2d_58,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            relu_18,
            conv2d_59,
            full_18,
            floor_17,
            divide_17,
            multiply_17,
            add_18,
            assign_37,
            split_38,
            split_39,
            conv2d_60,
            assign_38,
            concat_19,
            conv2d_61,
            batch_norm__113,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            relu_19,
            conv2d_62,
            full_19,
            floor_18,
            divide_18,
            multiply_18,
            add_19,
            assign_39,
            split_40,
            split_41,
            conv2d_63,
            assign_40,
            concat_20,
            conv2d_64,
            batch_norm__118,
            batch_norm__119,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            relu_20,
            conv2d_65,
            full_20,
            floor_19,
            divide_19,
            multiply_19,
            add_20,
            assign_41,
            split_42,
            split_43,
            conv2d_66,
            assign_42,
            concat_21,
            conv2d_67,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__126,
            batch_norm__127,
            relu_21,
            conv2d_68,
            full_21,
            floor_20,
            divide_20,
            multiply_20,
            add_21,
            assign_43,
            split_44,
            split_45,
            conv2d_69,
            assign_44,
            concat_22,
            conv2d_70,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            batch_norm__132,
            relu_22,
            conv2d_71,
            full_22,
            floor_21,
            divide_21,
            multiply_21,
            add_22,
            assign_45,
            split_46,
            split_47,
            conv2d_72,
            assign_46,
            concat_23,
            conv2d_73,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            relu_23,
            conv2d_74,
            full_23,
            floor_22,
            divide_22,
            multiply_22,
            add_23,
            assign_47,
            split_48,
            split_49,
            conv2d_75,
            assign_48,
            concat_24,
            conv2d_76,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            relu_24,
            conv2d_77,
            full_24,
            floor_23,
            divide_23,
            multiply_23,
            add_24,
            conv2d_78,
            batch_norm__143,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            assign_49,
            split_50,
            split_51,
            conv2d_79,
            assign_50,
            concat_25,
            conv2d_80,
            batch_norm__149,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            relu_25,
            conv2d_81,
            full_25,
            floor_24,
            divide_24,
            multiply_24,
            add_25,
            assign_51,
            split_52,
            split_53,
            conv2d_82,
            assign_52,
            concat_26,
            conv2d_83,
            batch_norm__154,
            batch_norm__155,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            relu_26,
            conv2d_84,
            full_26,
            floor_25,
            divide_25,
            multiply_25,
            add_26,
            assign_53,
            split_54,
            split_55,
            conv2d_85,
            assign_54,
            concat_27,
            conv2d_86,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            relu_27,
            conv2d_87,
            full_27,
            floor_26,
            divide_26,
            multiply_26,
            add_27,
            full_int_array_0,
            pool2d_0,
            relu_28,
            flatten_0,
            matmul_0,
            add_28,
        )
