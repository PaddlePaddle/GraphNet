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
        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x3x224x224xf32, 144x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_218, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_218

        # pd_op.batch_norm_: (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32, -1xui8) <- (-1x144x56x56xf32, 144xf32, 144xf32, 144xf32, 144xf32)
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
        del parameter_214, parameter_215, parameter_216, parameter_217

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [36, 108]

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

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_49 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_50 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_51 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_52 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_53 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_54 = full_0

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_213, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_213

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_0 = [conv2d_1, split_2]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_212, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_212

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
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
        del parameter_208, parameter_209, parameter_210, parameter_211

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_207, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_207

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(add_1, full_int_array_0, full_0)

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            split_4, parameter_206, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_206

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_1 = [conv2d_4, split_5]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_1, parameter_205, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_205

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
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
        del parameter_201, parameter_202, parameter_203, parameter_204

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_1, parameter_200, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_200

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [],
            float("0.992593"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x144x56x56xf32)
        shape64_0 = paddle._C_ops.shape64(conv2d_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [slice_0, full_2, full_2, full_2]
        del slice_0

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            stack_0,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_0

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_2 = paddle._C_ops.add(full_1, uniform_0)
        del uniform_0

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_2)
        del add_2

        # pd_op.divide: (-1x144x56x56xf32) <- (-1x144x56x56xf32, xf32)
        divide_0 = paddle._C_ops.divide(conv2d_6, full_1)

        # pd_op.multiply: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_3 = paddle._C_ops.add(add_1, multiply_0)

        # pd_op.split: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x144x56x56xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_3, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (-1x36x56x56xf32, -1x108x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (-1x36x56x56xf32) <- (-1x36x56x56xf32, 36x36x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            split_7, parameter_199, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_199

        # builtin.combine: ([-1x36x56x56xf32, -1x108x56x56xf32]) <- (-1x36x56x56xf32, -1x108x56x56xf32)
        combine_3 = [conv2d_7, split_8]

        # pd_op.concat: (-1x144x56x56xf32) <- ([-1x36x56x56xf32, -1x108x56x56xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (-1x288x56x56xf32) <- (-1x144x56x56xf32, 288x144x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_2, parameter_198, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_198

        # pd_op.batch_norm_: (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x56x56xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
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
        del parameter_194, parameter_195, parameter_196, parameter_197

        # pd_op.relu: (-1x288x56x56xf32) <- (-1x288x56x56xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x144x56x56xf32) <- (-1x288x56x56xf32, 144x288x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_2, parameter_193, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_193

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.985185"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x144x56x56xf32)
        shape64_1 = paddle._C_ops.shape64(conv2d_9)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_1

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [slice_1, full_2, full_2, full_2]
        del slice_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            stack_1,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_1

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_4 = paddle._C_ops.add(full_5, uniform_1)
        del uniform_1

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_4)
        del add_4

        # pd_op.divide: (-1x144x56x56xf32) <- (-1x144x56x56xf32, xf32)
        divide_1 = paddle._C_ops.divide(conv2d_9, full_5)

        # pd_op.multiply: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (-1x144x56x56xf32) <- (-1x144x56x56xf32, -1x144x56x56xf32)
        add_5 = paddle._C_ops.add(add_3, multiply_1)

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x144x56x56xf32, 288x144x2x2xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            add_5, parameter_192, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_192

        # pd_op.batch_norm_: (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32, -1xui8) <- (-1x288x28x28xf32, 288xf32, 288xf32, 288xf32, 288xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
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
        del parameter_188, parameter_189, parameter_190, parameter_191

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [72, 216]

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__24, full_int_array_3, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            split_10, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_5 = [conv2d_11, split_11]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            concat_3, parameter_186, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_186

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
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
        del parameter_182, parameter_183, parameter_184, parameter_185

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_3, parameter_181, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_181

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.977778"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        shape64_2 = paddle._C_ops.shape64(conv2d_13)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_6 = [slice_2, full_2, full_2, full_2]
        del slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            stack_2,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_2

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_6 = paddle._C_ops.add(full_6, uniform_2)
        del uniform_2

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_6)
        del add_6

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        divide_2 = paddle._C_ops.divide(conv2d_13, full_6)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_7 = paddle._C_ops.add(batch_norm__24, multiply_2)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_7, full_int_array_3, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            split_13, parameter_180, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_180

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_7 = [conv2d_14, split_14]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_4, parameter_179, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_179

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
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
        del parameter_175, parameter_176, parameter_177, parameter_178

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_4, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_174

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.97037"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        shape64_3 = paddle._C_ops.shape64(conv2d_16)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            shape64_3, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_3

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_8 = [slice_3, full_2, full_2, full_2]
        del slice_3

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_8, 0)
        del combine_8

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            stack_3,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_3

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_8 = paddle._C_ops.add(full_7, uniform_3)
        del uniform_3

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_8)
        del add_8

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        divide_3 = paddle._C_ops.divide(conv2d_16, full_7)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_9 = paddle._C_ops.add(add_7, multiply_3)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_9, full_int_array_3, full_0)

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            split_16, parameter_173, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_173

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_9 = [conv2d_17, split_17]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            concat_5, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_172

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
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
        del parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_5, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.962963"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        shape64_4 = paddle._C_ops.shape64(conv2d_19)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            shape64_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_10 = [slice_4, full_2, full_2, full_2]
        del slice_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_10, 0)
        del combine_10

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            stack_4,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_4

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_10 = paddle._C_ops.add(full_8, uniform_4)
        del uniform_4

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_10)
        del add_10

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        divide_4 = paddle._C_ops.divide(conv2d_19, full_8)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_11 = paddle._C_ops.add(add_9, multiply_4)

        # pd_op.split: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x288x28x28xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_11, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (-1x72x28x28xf32, -1x216x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (-1x72x28x28xf32) <- (-1x72x28x28xf32, 72x72x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            split_19, parameter_166, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_166

        # builtin.combine: ([-1x72x28x28xf32, -1x216x28x28xf32]) <- (-1x72x28x28xf32, -1x216x28x28xf32)
        combine_11 = [conv2d_20, split_20]

        # pd_op.concat: (-1x288x28x28xf32) <- ([-1x72x28x28xf32, -1x216x28x28xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (-1x576x28x28xf32) <- (-1x288x28x28xf32, 576x288x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            concat_6, parameter_165, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_165

        # pd_op.batch_norm_: (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x28x28xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
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
        del parameter_161, parameter_162, parameter_163, parameter_164

        # pd_op.relu: (-1x576x28x28xf32) <- (-1x576x28x28xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x288x28x28xf32) <- (-1x576x28x28xf32, 288x576x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_6, parameter_160, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_160

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.955556"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x288x28x28xf32)
        shape64_5 = paddle._C_ops.shape64(conv2d_22)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            shape64_5, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_5

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_12 = [slice_5, full_2, full_2, full_2]
        del slice_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            stack_5,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_5

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_12 = paddle._C_ops.add(full_9, uniform_5)
        del uniform_5

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_12)
        del add_12

        # pd_op.divide: (-1x288x28x28xf32) <- (-1x288x28x28xf32, xf32)
        divide_5 = paddle._C_ops.divide(conv2d_22, full_9)

        # pd_op.multiply: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (-1x288x28x28xf32) <- (-1x288x28x28xf32, -1x288x28x28xf32)
        add_13 = paddle._C_ops.add(add_11, multiply_5)

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x288x28x28xf32, 576x288x2x2xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            add_13, parameter_159, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_159

        # pd_op.batch_norm_: (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32, -1xui8) <- (-1x576x14x14xf32, 576xf32, 576xf32, 576xf32, 576xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
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
        del parameter_155, parameter_156, parameter_157, parameter_158

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [144, 432]

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(batch_norm__54, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_154, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_154

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_13 = [conv2d_24, split_23]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_13, full_0)
        del combine_13

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_153, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_153

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
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
        del parameter_149, parameter_150, parameter_151, parameter_152

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_148, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_148

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.948148"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_6 = paddle._C_ops.shape64(conv2d_26)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            shape64_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_14 = [slice_6, full_2, full_2, full_2]
        del slice_6

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_6 = paddle._C_ops.stack(combine_14, 0)
        del combine_14

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            stack_6,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_6

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_14 = paddle._C_ops.add(full_10, uniform_6)
        del uniform_6

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_14)
        del add_14

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(conv2d_26, full_10)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_15 = paddle._C_ops.add(batch_norm__54, multiply_6)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_15, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_147, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_15 = [conv2d_27, split_26]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_15, full_0)
        del combine_15

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_146, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_146

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
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
        del parameter_142, parameter_143, parameter_144, parameter_145

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_141, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_141

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.940741"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_7 = paddle._C_ops.shape64(conv2d_29)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            shape64_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_7

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_16 = [slice_7, full_2, full_2, full_2]
        del slice_7

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_16, 0)
        del combine_16

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            stack_7,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_7

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_16 = paddle._C_ops.add(full_11, uniform_7)
        del uniform_7

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_16)
        del add_16

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(conv2d_29, full_11)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_17 = paddle._C_ops.add(add_15, multiply_7)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_17, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_140

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_17 = [conv2d_30, split_29]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_17, full_0)
        del combine_17

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_139

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
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
        del parameter_135, parameter_136, parameter_137, parameter_138

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_134

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.933333"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_8 = paddle._C_ops.shape64(conv2d_32)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_8, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_18 = [slice_8, full_2, full_2, full_2]
        del slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_8 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            stack_8,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_8

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_18 = paddle._C_ops.add(full_12, uniform_8)
        del uniform_8

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_18)
        del add_18

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(conv2d_32, full_12)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_19 = paddle._C_ops.add(add_17, multiply_8)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_19, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_133, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_133

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_19 = [conv2d_33, split_32]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_19, full_0)
        del combine_19

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_132, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_132

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
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
        del parameter_128, parameter_129, parameter_130, parameter_131

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_127

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.925926"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_9 = paddle._C_ops.shape64(conv2d_35)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_9, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_9

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_20 = [slice_9, full_2, full_2, full_2]
        del slice_9

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_20, 0)
        del combine_20

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            stack_9,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_9

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_20 = paddle._C_ops.add(full_13, uniform_9)
        del uniform_9

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_20)
        del add_20

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(conv2d_35, full_13)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_21 = paddle._C_ops.add(add_19, multiply_9)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(add_21, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            split_34, parameter_126, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_126

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_21 = [conv2d_36, split_35]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_21, full_0)
        del combine_21

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            concat_11, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_125

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
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
        del parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_11, parameter_120, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.918519"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_10 = paddle._C_ops.shape64(conv2d_38)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_10, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_10

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_22 = [slice_10, full_2, full_2, full_2]
        del slice_10

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_10 = paddle._C_ops.stack(combine_22, 0)
        del combine_22

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            stack_10,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_10

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_14, uniform_10)
        del uniform_10

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_22)
        del add_22

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_10 = paddle._C_ops.divide(conv2d_38, full_14)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_23 = paddle._C_ops.add(add_21, multiply_10)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_23, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            split_37, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_119

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_23 = [conv2d_39, split_38]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_23, full_0)
        del combine_23

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_12, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_118

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
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
        del parameter_114, parameter_115, parameter_116, parameter_117

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_12, parameter_113, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_113

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.911111"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_11 = paddle._C_ops.shape64(conv2d_41)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_11, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_11

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_24 = [slice_11, full_2, full_2, full_2]
        del slice_11

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_11 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            stack_11,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_11

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_24 = paddle._C_ops.add(full_15, uniform_11)
        del uniform_11

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_24)
        del add_24

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_11 = paddle._C_ops.divide(conv2d_41, full_15)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_25 = paddle._C_ops.add(add_23, multiply_11)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_39 = paddle._C_ops.split(add_25, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_40,
            split_41,
        ) = split_39
        del split_39

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            split_40, parameter_112, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_112

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_25 = [conv2d_42, split_41]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_25, full_0)
        del combine_25

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            concat_13, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_111

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
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
        del parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_13, parameter_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_106

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [],
            float("0.903704"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_12 = paddle._C_ops.shape64(conv2d_44)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_12

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_26 = [slice_12, full_2, full_2, full_2]
        del slice_12

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_12 = paddle._C_ops.stack(combine_26, 0)
        del combine_26

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            stack_12,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_12

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_26 = paddle._C_ops.add(full_16, uniform_12)
        del uniform_12

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_26)
        del add_26

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_12 = paddle._C_ops.divide(conv2d_44, full_16)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_27 = paddle._C_ops.add(add_25, multiply_12)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(add_27, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_43,
            split_44,
        ) = split_42
        del split_42

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            split_43, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_27 = [conv2d_45, split_44]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_27, full_0)
        del combine_27

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            concat_14, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_104

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
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
        del parameter_100, parameter_101, parameter_102, parameter_103

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_14, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_99

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [],
            float("0.896296"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_13 = paddle._C_ops.shape64(conv2d_47)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            shape64_13, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_13

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_28 = [slice_13, full_2, full_2, full_2]
        del slice_13

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_13 = paddle._C_ops.stack(combine_28, 0)
        del combine_28

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            stack_13,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_13

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_28 = paddle._C_ops.add(full_17, uniform_13)
        del uniform_13

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_28)
        del add_28

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_13 = paddle._C_ops.divide(conv2d_47, full_17)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_29 = paddle._C_ops.add(add_27, multiply_13)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_45 = paddle._C_ops.split(add_29, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_46,
            split_47,
        ) = split_45
        del split_45

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            split_46, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_29 = [conv2d_48, split_47]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_29, full_0)
        del combine_29

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            concat_15, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_97

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
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
        del parameter_93, parameter_94, parameter_95, parameter_96

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_15, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92

        # pd_op.full: (xf32) <- ()
        full_18 = paddle._C_ops.full(
            [],
            float("0.888889"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_14 = paddle._C_ops.shape64(conv2d_50)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            shape64_14, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_14

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_30 = [slice_14, full_2, full_2, full_2]
        del slice_14

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_14 = paddle._C_ops.stack(combine_30, 0)
        del combine_30

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            stack_14,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_14

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_30 = paddle._C_ops.add(full_18, uniform_14)
        del uniform_14

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_30)
        del add_30

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_14 = paddle._C_ops.divide(conv2d_50, full_18)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_31 = paddle._C_ops.add(add_29, multiply_14)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_48 = paddle._C_ops.split(add_31, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_49,
            split_50,
        ) = split_48
        del split_48

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            split_49, parameter_91, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_91

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_31 = [conv2d_51, split_50]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_31, full_0)
        del combine_31

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            concat_16, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
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
        del parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            relu_16, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_85

        # pd_op.full: (xf32) <- ()
        full_19 = paddle._C_ops.full(
            [],
            float("0.881481"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_15 = paddle._C_ops.shape64(conv2d_53)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            shape64_15, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_15

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_32 = [slice_15, full_2, full_2, full_2]
        del slice_15

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_15 = paddle._C_ops.stack(combine_32, 0)
        del combine_32

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            stack_15,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_15

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_32 = paddle._C_ops.add(full_19, uniform_15)
        del uniform_15

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_32)
        del add_32

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_15 = paddle._C_ops.divide(conv2d_53, full_19)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_33 = paddle._C_ops.add(add_31, multiply_15)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_51 = paddle._C_ops.split(add_33, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_52,
            split_53,
        ) = split_51
        del split_51

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            split_52, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_33 = [conv2d_54, split_53]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_33, full_0)
        del combine_33

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            concat_17, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
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
        del parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            relu_17, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78

        # pd_op.full: (xf32) <- ()
        full_20 = paddle._C_ops.full(
            [],
            float("0.874074"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_16 = paddle._C_ops.shape64(conv2d_56)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            shape64_16, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_16

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_34 = [slice_16, full_2, full_2, full_2]
        del slice_16

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_16 = paddle._C_ops.stack(combine_34, 0)
        del combine_34

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            stack_16,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_16

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_34 = paddle._C_ops.add(full_20, uniform_16)
        del uniform_16

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_34)
        del add_34

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_16 = paddle._C_ops.divide(conv2d_56, full_20)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_35 = paddle._C_ops.add(add_33, multiply_16)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_54 = paddle._C_ops.split(add_35, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_55,
            split_56,
        ) = split_54
        del split_54

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            split_55, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_35 = [conv2d_57, split_56]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_35, full_0)
        del combine_35

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            concat_18, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_76

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
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
        del parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(
            relu_18, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.full: (xf32) <- ()
        full_21 = paddle._C_ops.full(
            [],
            float("0.866667"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_17 = paddle._C_ops.shape64(conv2d_59)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            shape64_17, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_17

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_36 = [slice_17, full_2, full_2, full_2]
        del slice_17

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_17 = paddle._C_ops.stack(combine_36, 0)
        del combine_36

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_17 = paddle._C_ops.uniform(
            stack_17,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_17

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_36 = paddle._C_ops.add(full_21, uniform_17)
        del uniform_17

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_17 = paddle._C_ops.floor(add_36)
        del add_36

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_17 = paddle._C_ops.divide(conv2d_59, full_21)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_17 = paddle._C_ops.multiply(divide_17, floor_17)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_37 = paddle._C_ops.add(add_35, multiply_17)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_57 = paddle._C_ops.split(add_37, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_58,
            split_59,
        ) = split_57
        del split_57

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_60 = paddle._C_ops.conv2d(
            split_58, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_37 = [conv2d_60, split_59]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_37, full_0)
        del combine_37

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(
            concat_19, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
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
        del parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(
            relu_19, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.full: (xf32) <- ()
        full_22 = paddle._C_ops.full(
            [],
            float("0.859259"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_18 = paddle._C_ops.shape64(conv2d_62)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            shape64_18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_18

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_38 = [slice_18, full_2, full_2, full_2]
        del slice_18

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_18 = paddle._C_ops.stack(combine_38, 0)
        del combine_38

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_18 = paddle._C_ops.uniform(
            stack_18,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_18

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_38 = paddle._C_ops.add(full_22, uniform_18)
        del uniform_18

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_18 = paddle._C_ops.floor(add_38)
        del add_38

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_18 = paddle._C_ops.divide(conv2d_62, full_22)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_18 = paddle._C_ops.multiply(divide_18, floor_18)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_39 = paddle._C_ops.add(add_37, multiply_18)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_60 = paddle._C_ops.split(add_39, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_61,
            split_62,
        ) = split_60
        del split_60

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(
            split_61, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_39 = [conv2d_63, split_62]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_39, full_0)
        del combine_39

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(
            concat_20, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
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
        del parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(
            relu_20, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.full: (xf32) <- ()
        full_23 = paddle._C_ops.full(
            [],
            float("0.851852"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_19 = paddle._C_ops.shape64(conv2d_65)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            shape64_19, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_19

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_40 = [slice_19, full_2, full_2, full_2]
        del slice_19

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_19 = paddle._C_ops.stack(combine_40, 0)
        del combine_40

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_19 = paddle._C_ops.uniform(
            stack_19,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_19

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_40 = paddle._C_ops.add(full_23, uniform_19)
        del uniform_19

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_19 = paddle._C_ops.floor(add_40)
        del add_40

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_19 = paddle._C_ops.divide(conv2d_65, full_23)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_19 = paddle._C_ops.multiply(divide_19, floor_19)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_41 = paddle._C_ops.add(add_39, multiply_19)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_63 = paddle._C_ops.split(add_41, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_64,
            split_65,
        ) = split_63
        del split_63

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(
            split_64, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_41 = [conv2d_66, split_65]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_41, full_0)
        del combine_41

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(
            concat_21, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
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
        del parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(
            relu_21, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50

        # pd_op.full: (xf32) <- ()
        full_24 = paddle._C_ops.full(
            [],
            float("0.844444"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_20 = paddle._C_ops.shape64(conv2d_68)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            shape64_20, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_20

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_42 = [slice_20, full_2, full_2, full_2]
        del slice_20

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_20 = paddle._C_ops.stack(combine_42, 0)
        del combine_42

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_20 = paddle._C_ops.uniform(
            stack_20,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_20

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_42 = paddle._C_ops.add(full_24, uniform_20)
        del uniform_20

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_20 = paddle._C_ops.floor(add_42)
        del add_42

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_20 = paddle._C_ops.divide(conv2d_68, full_24)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_20 = paddle._C_ops.multiply(divide_20, floor_20)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_43 = paddle._C_ops.add(add_41, multiply_20)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_66 = paddle._C_ops.split(add_43, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_67,
            split_68,
        ) = split_66
        del split_66

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(
            split_67, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_43 = [conv2d_69, split_68]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_43, full_0)
        del combine_43

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(
            concat_22, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_48

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
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
        del parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__150)
        del batch_norm__150

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(
            relu_22, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43

        # pd_op.full: (xf32) <- ()
        full_25 = paddle._C_ops.full(
            [],
            float("0.837037"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_21 = paddle._C_ops.shape64(conv2d_71)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            shape64_21, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_21

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_44 = [slice_21, full_2, full_2, full_2]
        del slice_21

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_21 = paddle._C_ops.stack(combine_44, 0)
        del combine_44

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_21 = paddle._C_ops.uniform(
            stack_21,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_21

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_44 = paddle._C_ops.add(full_25, uniform_21)
        del uniform_21

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_21 = paddle._C_ops.floor(add_44)
        del add_44

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_21 = paddle._C_ops.divide(conv2d_71, full_25)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_21 = paddle._C_ops.multiply(divide_21, floor_21)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_45 = paddle._C_ops.add(add_43, multiply_21)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_69 = paddle._C_ops.split(add_45, full_int_array_4, full_0)

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_70,
            split_71,
        ) = split_69
        del split_69

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(
            split_70, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_45 = [conv2d_72, split_71]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_45, full_0)
        del combine_45

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(
            concat_23, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
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
        del parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(
            relu_23, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.full: (xf32) <- ()
        full_26 = paddle._C_ops.full(
            [],
            float("0.82963"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_22 = paddle._C_ops.shape64(conv2d_74)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            shape64_22, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_22

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_46 = [slice_22, full_2, full_2, full_2]
        del slice_22

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_22 = paddle._C_ops.stack(combine_46, 0)
        del combine_46

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_22 = paddle._C_ops.uniform(
            stack_22,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_22

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_46 = paddle._C_ops.add(full_26, uniform_22)
        del uniform_22

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_22 = paddle._C_ops.floor(add_46)
        del add_46

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_22 = paddle._C_ops.divide(conv2d_74, full_26)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_22 = paddle._C_ops.multiply(divide_22, floor_22)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_47 = paddle._C_ops.add(add_45, multiply_22)

        # pd_op.split: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x576x14x14xf32, 2xi64, 1xi32)
        split_72 = paddle._C_ops.split(add_47, full_int_array_4, full_0)
        del full_int_array_4

        # builtin.split: (-1x144x14x14xf32, -1x432x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32])
        (
            split_73,
            split_74,
        ) = split_72
        del split_72

        # pd_op.conv2d: (-1x144x14x14xf32) <- (-1x144x14x14xf32, 144x144x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(
            split_73, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # builtin.combine: ([-1x144x14x14xf32, -1x432x14x14xf32]) <- (-1x144x14x14xf32, -1x432x14x14xf32)
        combine_47 = [conv2d_75, split_74]

        # pd_op.concat: (-1x576x14x14xf32) <- ([-1x144x14x14xf32, -1x432x14x14xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_47, full_0)
        del combine_47

        # pd_op.conv2d: (-1x1152x14x14xf32) <- (-1x576x14x14xf32, 1152x576x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(
            concat_24, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x14x14xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
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
        del parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (-1x1152x14x14xf32) <- (-1x1152x14x14xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__162)
        del batch_norm__162

        # pd_op.conv2d: (-1x576x14x14xf32) <- (-1x1152x14x14xf32, 576x1152x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(
            relu_24, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.full: (xf32) <- ()
        full_27 = paddle._C_ops.full(
            [],
            float("0.822222"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x576x14x14xf32)
        shape64_23 = paddle._C_ops.shape64(conv2d_77)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(
            shape64_23, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_23

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_48 = [slice_23, full_2, full_2, full_2]
        del slice_23

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_23 = paddle._C_ops.stack(combine_48, 0)
        del combine_48

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_23 = paddle._C_ops.uniform(
            stack_23,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_23

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_48 = paddle._C_ops.add(full_27, uniform_23)
        del uniform_23

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_23 = paddle._C_ops.floor(add_48)
        del add_48

        # pd_op.divide: (-1x576x14x14xf32) <- (-1x576x14x14xf32, xf32)
        divide_23 = paddle._C_ops.divide(conv2d_77, full_27)

        # pd_op.multiply: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x1x1x1xf32)
        multiply_23 = paddle._C_ops.multiply(divide_23, floor_23)

        # pd_op.add: (-1x576x14x14xf32) <- (-1x576x14x14xf32, -1x576x14x14xf32)
        add_49 = paddle._C_ops.add(add_47, multiply_23)

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x576x14x14xf32, 1152x576x2x2xf32)
        conv2d_78 = paddle._C_ops.conv2d(
            add_49, parameter_28, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # pd_op.batch_norm_: (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32, -1xui8) <- (-1x1152x7x7xf32, 1152xf32, 1152xf32, 1152xf32, 1152xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
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
        del parameter_24, parameter_25, parameter_26, parameter_27

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [288, 864]

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_75 = paddle._C_ops.split(batch_norm__168, full_int_array_5, full_0)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_76,
            split_77,
        ) = split_75
        del split_75

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(
            split_76, parameter_23, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_23

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_49 = [conv2d_79, split_77]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_49, full_0)
        del combine_49

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(
            concat_25, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
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
        del parameter_18, parameter_19, parameter_20, parameter_21

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__174)
        del batch_norm__174

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(
            relu_25, parameter_17, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_17

        # pd_op.full: (xf32) <- ()
        full_28 = paddle._C_ops.full(
            [],
            float("0.814815"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        shape64_24 = paddle._C_ops.shape64(conv2d_81)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            shape64_24, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_24

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_50 = [slice_24, full_2, full_2, full_2]
        del slice_24

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_24 = paddle._C_ops.stack(combine_50, 0)
        del combine_50

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_24 = paddle._C_ops.uniform(
            stack_24,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_24

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_50 = paddle._C_ops.add(full_28, uniform_24)
        del uniform_24

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_24 = paddle._C_ops.floor(add_50)
        del add_50

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        divide_24 = paddle._C_ops.divide(conv2d_81, full_28)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        multiply_24 = paddle._C_ops.multiply(divide_24, floor_24)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_51 = paddle._C_ops.add(batch_norm__168, multiply_24)

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_78 = paddle._C_ops.split(add_51, full_int_array_5, full_0)

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_79,
            split_80,
        ) = split_78
        del split_78

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(
            split_79, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_51 = [conv2d_82, split_80]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_51, full_0)
        del combine_51

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(
            concat_26, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
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
        del parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(
            relu_26, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10

        # pd_op.full: (xf32) <- ()
        full_29 = paddle._C_ops.full(
            [],
            float("0.807407"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        shape64_25 = paddle._C_ops.shape64(conv2d_84)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            shape64_25, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del shape64_25

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_52 = [slice_25, full_2, full_2, full_2]
        del slice_25

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_25 = paddle._C_ops.stack(combine_52, 0)
        del combine_52

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_25 = paddle._C_ops.uniform(
            stack_25,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del stack_25

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_52 = paddle._C_ops.add(full_29, uniform_25)
        del uniform_25

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_25 = paddle._C_ops.floor(add_52)
        del add_52

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        divide_25 = paddle._C_ops.divide(conv2d_84, full_29)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        multiply_25 = paddle._C_ops.multiply(divide_25, floor_25)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_53 = paddle._C_ops.add(add_51, multiply_25)

        # pd_op.split: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x1152x7x7xf32, 2xi64, 1xi32)
        split_81 = paddle._C_ops.split(add_53, full_int_array_5, full_0)
        del full_int_array_5

        # builtin.split: (-1x288x7x7xf32, -1x864x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32])
        (
            split_82,
            split_83,
        ) = split_81
        del split_81

        # pd_op.conv2d: (-1x288x7x7xf32) <- (-1x288x7x7xf32, 288x288x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(
            split_82, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # builtin.combine: ([-1x288x7x7xf32, -1x864x7x7xf32]) <- (-1x288x7x7xf32, -1x864x7x7xf32)
        combine_53 = [conv2d_85, split_83]

        # pd_op.concat: (-1x1152x7x7xf32) <- ([-1x288x7x7xf32, -1x864x7x7xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_53, full_0)
        del combine_53

        # pd_op.conv2d: (-1x2304x7x7xf32) <- (-1x1152x7x7xf32, 2304x1152x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(
            concat_27, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.batch_norm_: (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32, -1xui8) <- (-1x2304x7x7xf32, 2304xf32, 2304xf32, 2304xf32, 2304xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
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
        del parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.relu: (-1x2304x7x7xf32) <- (-1x2304x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (-1x1152x7x7xf32) <- (-1x2304x7x7xf32, 1152x2304x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(
            relu_27, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.full: (xf32) <- ()
        full_30 = paddle._C_ops.full(
            [], float("0.8"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.shape64: (4xi64) <- (-1x1152x7x7xf32)
        shape64_26 = paddle._C_ops.shape64(conv2d_87)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            shape64_26, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_1, full_int_array_2, shape64_26

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_54 = [slice_26, full_2, full_2, full_2]
        del full_2, slice_26

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_26 = paddle._C_ops.stack(combine_54, 0)
        del combine_54

        # pd_op.uniform: (-1x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_26 = paddle._C_ops.uniform(
            stack_26,
            paddle.float32,
            full_3,
            full_4,
            0,
            paddle.framework._current_expected_place(),
        )
        del full_3, full_4, stack_26

        # pd_op.add: (-1x1x1x1xf32) <- (xf32, -1x1x1x1xf32)
        add_54 = paddle._C_ops.add(full_30, uniform_26)
        del uniform_26

        # pd_op.floor: (-1x1x1x1xf32) <- (-1x1x1x1xf32)
        floor_26 = paddle._C_ops.floor(add_54)
        del add_54

        # pd_op.divide: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, xf32)
        divide_26 = paddle._C_ops.divide(conv2d_87, full_30)

        # pd_op.multiply: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1x1x1xf32)
        multiply_26 = paddle._C_ops.multiply(divide_26, floor_26)

        # pd_op.add: (-1x1152x7x7xf32) <- (-1x1152x7x7xf32, -1x1152x7x7xf32)
        add_55 = paddle._C_ops.add(add_53, multiply_26)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.pool2d: (-1x1152x1x1xf32) <- (-1x1152x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_55,
            full_int_array_6,
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

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1152x1x1xf32, 1280x1152x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_28 = paddle._C_ops.relu(conv2d_88)
        del conv2d_88

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_28, 1, 3)

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_1,
            add_11,
            add_13,
            add_15,
            add_17,
            add_19,
            add_21,
            add_23,
            add_25,
            add_27,
            add_29,
            add_3,
            add_31,
            add_33,
            add_35,
            add_37,
            add_39,
            add_41,
            add_43,
            add_45,
            add_47,
            add_49,
            add_5,
            add_51,
            add_53,
            add_55,
            add_7,
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
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__109,
            batch_norm__11,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__13,
            batch_norm__130,
            batch_norm__131,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__139,
            batch_norm__14,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__15,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__16,
            batch_norm__160,
            batch_norm__161,
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
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__19,
            batch_norm__190,
            batch_norm__191,
            batch_norm__2,
            batch_norm__20,
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
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__4,
            batch_norm__40,
            batch_norm__41,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
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
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__7,
            batch_norm__70,
            batch_norm__71,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__79,
            batch_norm__8,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            batch_norm__9,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            concat_0,
            concat_1,
            concat_10,
            concat_11,
            concat_12,
            concat_13,
            concat_14,
            concat_15,
            concat_16,
            concat_17,
            concat_18,
            concat_19,
            concat_2,
            concat_20,
            concat_21,
            concat_22,
            concat_23,
            concat_24,
            concat_25,
            concat_26,
            concat_27,
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
            conv2d_35,
            conv2d_36,
            conv2d_37,
            conv2d_38,
            conv2d_39,
            conv2d_4,
            conv2d_40,
            conv2d_41,
            conv2d_42,
            conv2d_43,
            conv2d_44,
            conv2d_45,
            conv2d_46,
            conv2d_47,
            conv2d_48,
            conv2d_49,
            conv2d_5,
            conv2d_50,
            conv2d_51,
            conv2d_52,
            conv2d_53,
            conv2d_54,
            conv2d_55,
            conv2d_56,
            conv2d_57,
            conv2d_58,
            conv2d_59,
            conv2d_6,
            conv2d_60,
            conv2d_61,
            conv2d_62,
            conv2d_63,
            conv2d_64,
            conv2d_65,
            conv2d_66,
            conv2d_67,
            conv2d_68,
            conv2d_69,
            conv2d_7,
            conv2d_70,
            conv2d_71,
            conv2d_72,
            conv2d_73,
            conv2d_74,
            conv2d_75,
            conv2d_76,
            conv2d_77,
            conv2d_78,
            conv2d_79,
            conv2d_8,
            conv2d_80,
            conv2d_81,
            conv2d_82,
            conv2d_83,
            conv2d_84,
            conv2d_85,
            conv2d_86,
            conv2d_87,
            conv2d_9,
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
            divide_22,
            divide_23,
            divide_24,
            divide_25,
            divide_26,
            divide_3,
            divide_4,
            divide_5,
            divide_6,
            divide_7,
            divide_8,
            divide_9,
            flatten_0,
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
            floor_22,
            floor_23,
            floor_24,
            floor_25,
            floor_26,
            floor_3,
            floor_4,
            floor_5,
            floor_6,
            floor_7,
            floor_8,
            floor_9,
            full_0,
            full_1,
            full_10,
            full_11,
            full_12,
            full_13,
            full_14,
            full_15,
            full_16,
            full_17,
            full_18,
            full_19,
            full_20,
            full_21,
            full_22,
            full_23,
            full_24,
            full_25,
            full_26,
            full_27,
            full_28,
            full_29,
            full_30,
            full_5,
            full_6,
            full_7,
            full_8,
            full_9,
            full_int_array_6,
            matmul_0,
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
            multiply_22,
            multiply_23,
            multiply_24,
            multiply_25,
            multiply_26,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
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
            relu_3,
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
            split_1,
            split_10,
            split_11,
            split_13,
            split_14,
            split_16,
            split_17,
            split_19,
            split_2,
            split_20,
            split_22,
            split_23,
            split_25,
            split_26,
            split_28,
            split_29,
            split_31,
            split_32,
            split_34,
            split_35,
            split_37,
            split_38,
            split_4,
            split_40,
            split_41,
            split_43,
            split_44,
            split_46,
            split_47,
            split_49,
            split_5,
            split_50,
            split_52,
            split_53,
            split_55,
            split_56,
            split_58,
            split_59,
            split_61,
            split_62,
            split_64,
            split_65,
            split_67,
            split_68,
            split_7,
            split_70,
            split_71,
            split_73,
            split_74,
            split_76,
            split_77,
            split_79,
            split_8,
            split_80,
            split_82,
            split_83,
        )

        return add_0
