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
        data_27,
        data_28,
        data_29,
        data_30,
        data_31,
        data_32,
        data_33,
        data_34,
        data_35,
        data_36,
        data_37,
        data_38,
        data_39,
        data_40,
        data_41,
        data_42,
        data_43,
        data_44,
        data_45,
        data_46,
        data_47,
        data_48,
        data_49,
        data_50,
        data_51,
        data_52,
        data_53,
        data_54,
        data_55,
        data_56,
        data_57,
        data_58,
        data_59,
        data_60,
        data_61,
        data_62,
    ):
        # pd_op.conv2d: (64x24x112x112xf32) <- (64x3x224x224xf32, 24x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_62, parameter_212, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_62, parameter_212

        # pd_op.batch_norm_: (64x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (64x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.relu: (64x24x112x112xf32) <- (64x24x112x112xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.multiply: (64x24x112x112xf32) <- (1xf32, 64x24x112x112xf32)
        multiply_0 = paddle._C_ops.multiply(data_0, relu_0)
        del data_0

        # pd_op.add: (64x24x112x112xf32) <- (64x24x112x112xf32, 1xf32)
        add_1 = paddle._C_ops.add(multiply_0, data_1)
        del data_1

        # pd_op.conv2d: (64x12x112x112xf32) <- (64x24x112x112xf32, 12x24x2x2xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            add_1, parameter_207, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_207

        # pd_op.batch_norm_: (64x12x112x112xf32, 12xf32, 12xf32, 12xf32, 12xf32, -1xui8) <- (64x12x112x112xf32, 12xf32, 12xf32, 12xf32, 12xf32)
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
                parameter_206,
                parameter_205,
                parameter_204,
                parameter_203,
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
        del parameter_203, parameter_204, parameter_205, parameter_206

        # pd_op.relu: (64x12x112x112xf32) <- (64x12x112x112xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.multiply: (64x12x112x112xf32) <- (1xf32, 64x12x112x112xf32)
        multiply_1 = paddle._C_ops.multiply(data_12, relu_1)
        del data_12

        # pd_op.add: (64x12x112x112xf32) <- (64x12x112x112xf32, 1xf32)
        add_2 = paddle._C_ops.add(multiply_1, data_23)
        del data_23

        # pd_op.conv2d: (64x24x112x112xf32) <- (64x12x112x112xf32, 24x12x2x2xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            add_2, parameter_202, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_202

        # pd_op.batch_norm_: (64x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (64x24x112x112xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_201,
                parameter_200,
                parameter_199,
                parameter_198,
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
        del parameter_198, parameter_199, parameter_200, parameter_201

        # pd_op.relu: (64x24x112x112xf32) <- (64x24x112x112xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.multiply: (64x24x112x112xf32) <- (1xf32, 64x24x112x112xf32)
        multiply_2 = paddle._C_ops.multiply(data_34, relu_2)
        del data_34

        # pd_op.add: (64x24x112x112xf32) <- (64x24x112x112xf32, 1xf32)
        add_3 = paddle._C_ops.add(multiply_2, data_45)
        del data_45

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 2]

        # pd_op.pool2d: (64x24x112x112xf32) <- (64x24x112x112xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_1,
            full_int_array_0,
            [1, 1],
            [0, 0],
            True,
            True,
            "NCHW",
            "max",
            False,
            False,
            "SAME",
        )

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

        # builtin.combine: ([64x24x112x112xf32, 64x24x112x112xf32]) <- (64x24x112x112xf32, 64x24x112x112xf32)
        combine_0 = [pool2d_0, add_3]

        # pd_op.concat: (64x48x112x112xf32) <- ([64x24x112x112xf32, 64x24x112x112xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (64x24x56x56xf32) <- (64x48x112x112xf32, 24x48x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            concat_0, parameter_197, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_197

        # pd_op.batch_norm_: (64x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (64x24x56x56xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_196,
                parameter_195,
                parameter_194,
                parameter_193,
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
        del parameter_193, parameter_194, parameter_195, parameter_196

        # pd_op.relu: (64x24x56x56xf32) <- (64x24x56x56xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.multiply: (64x24x56x56xf32) <- (1xf32, 64x24x56x56xf32)
        multiply_3 = paddle._C_ops.multiply(data_56, relu_3)
        del data_56

        # pd_op.add: (64x24x56x56xf32) <- (64x24x56x56xf32, 1xf32)
        add_4 = paddle._C_ops.add(multiply_3, data_59)
        del data_59

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x24x56x56xf32, 32x24x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_4, parameter_192, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_192

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.multiply: (64x32x56x56xf32) <- (1xf32, 64x32x56x56xf32)
        multiply_4 = paddle._C_ops.multiply(data_60, relu_4)
        del data_60

        # pd_op.add: (64x32x56x56xf32) <- (64x32x56x56xf32, 1xf32)
        add_5 = paddle._C_ops.add(multiply_4, data_61)
        del data_61

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x32x56x56xf32, 32x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_5, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_186,
                parameter_185,
                parameter_184,
                parameter_183,
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
        del parameter_183, parameter_184, parameter_185, parameter_186

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.multiply: (64x32x56x56xf32) <- (1xf32, 64x32x56x56xf32)
        multiply_5 = paddle._C_ops.multiply(data_2, relu_5)
        del data_2

        # pd_op.add: (64x32x56x56xf32) <- (64x32x56x56xf32, 1xf32)
        add_6 = paddle._C_ops.add(multiply_5, data_3)
        del data_3

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x32x56x56xf32, 32x32x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            add_6, parameter_182, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_181,
                parameter_180,
                parameter_179,
                parameter_178,
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
        del parameter_178, parameter_179, parameter_180, parameter_181

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.multiply: (64x32x56x56xf32) <- (1xf32, 64x32x56x56xf32)
        multiply_6 = paddle._C_ops.multiply(data_4, relu_6)
        del data_4

        # pd_op.add: (64x32x56x56xf32) <- (64x32x56x56xf32, 1xf32)
        add_7 = paddle._C_ops.add(multiply_6, data_5)
        del data_5

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x32x56x56xf32, 32x32x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_7, parameter_177, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_177

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_176,
                parameter_175,
                parameter_174,
                parameter_173,
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
        del parameter_173, parameter_174, parameter_175, parameter_176

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.multiply: (64x32x56x56xf32) <- (1xf32, 64x32x56x56xf32)
        multiply_7 = paddle._C_ops.multiply(data_6, relu_7)
        del data_6

        # pd_op.add: (64x32x56x56xf32) <- (64x32x56x56xf32, 1xf32)
        add_8 = paddle._C_ops.add(multiply_7, data_7)
        del data_7

        # builtin.combine: ([64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32]) <- (64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32)
        combine_1 = [add_5, add_6, add_7, add_8]

        # pd_op.concat: (64x128x56x56xf32) <- ([64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32, 64x32x56x56xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (64x32x56x56xf32) <- (64x128x56x56xf32, 32x128x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_1, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_172

        # pd_op.batch_norm_: (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (64x32x56x56xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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

        # pd_op.relu: (64x32x56x56xf32) <- (64x32x56x56xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.multiply: (64x32x56x56xf32) <- (1xf32, 64x32x56x56xf32)
        multiply_8 = paddle._C_ops.multiply(data_8, relu_8)
        del data_8

        # pd_op.add: (64x32x56x56xf32) <- (64x32x56x56xf32, 1xf32)
        add_9 = paddle._C_ops.add(multiply_8, data_9)
        del data_9

        # pd_op.conv2d: (64x64x56x56xf32) <- (64x32x56x56xf32, 64x32x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            add_9, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167

        # pd_op.batch_norm_: (64x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_166,
                parameter_165,
                parameter_164,
                parameter_163,
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
        del parameter_163, parameter_164, parameter_165, parameter_166

        # pd_op.relu: (64x64x56x56xf32) <- (64x64x56x56xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.multiply: (64x64x56x56xf32) <- (1xf32, 64x64x56x56xf32)
        multiply_9 = paddle._C_ops.multiply(data_10, relu_9)
        del data_10

        # pd_op.add: (64x64x56x56xf32) <- (64x64x56x56xf32, 1xf32)
        add_10 = paddle._C_ops.add(multiply_9, data_11)
        del data_11

        # pd_op.depthwise_conv2d: (64x64x28x28xf32) <- (64x64x56x56xf32, 64x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            add_10, parameter_162, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_162

        # pd_op.batch_norm_: (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (64x64x28x28xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_161,
                parameter_160,
                parameter_159,
                parameter_158,
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
        del parameter_158, parameter_159, parameter_160, parameter_161

        # pd_op.conv2d: (64x48x28x28xf32) <- (64x64x28x28xf32, 48x64x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__60, parameter_157, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_157

        # pd_op.batch_norm_: (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_156,
                parameter_155,
                parameter_154,
                parameter_153,
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
        del parameter_153, parameter_154, parameter_155, parameter_156

        # pd_op.relu: (64x48x28x28xf32) <- (64x48x28x28xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.multiply: (64x48x28x28xf32) <- (1xf32, 64x48x28x28xf32)
        multiply_10 = paddle._C_ops.multiply(data_13, relu_10)
        del data_13

        # pd_op.add: (64x48x28x28xf32) <- (64x48x28x28xf32, 1xf32)
        add_11 = paddle._C_ops.add(multiply_10, data_14)
        del data_14

        # pd_op.conv2d: (64x48x28x28xf32) <- (64x48x28x28xf32, 48x48x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_11, parameter_152, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_152

        # pd_op.batch_norm_: (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_151,
                parameter_150,
                parameter_149,
                parameter_148,
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
        del parameter_148, parameter_149, parameter_150, parameter_151

        # pd_op.relu: (64x48x28x28xf32) <- (64x48x28x28xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.multiply: (64x48x28x28xf32) <- (1xf32, 64x48x28x28xf32)
        multiply_11 = paddle._C_ops.multiply(data_15, relu_11)
        del data_15

        # pd_op.add: (64x48x28x28xf32) <- (64x48x28x28xf32, 1xf32)
        add_12 = paddle._C_ops.add(multiply_11, data_16)
        del data_16

        # pd_op.conv2d: (64x48x28x28xf32) <- (64x48x28x28xf32, 48x48x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            add_12, parameter_147, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147

        # pd_op.batch_norm_: (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (64x48x28x28xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_146,
                parameter_145,
                parameter_144,
                parameter_143,
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
        del parameter_143, parameter_144, parameter_145, parameter_146

        # pd_op.relu: (64x48x28x28xf32) <- (64x48x28x28xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.multiply: (64x48x28x28xf32) <- (1xf32, 64x48x28x28xf32)
        multiply_12 = paddle._C_ops.multiply(data_17, relu_12)
        del data_17

        # pd_op.add: (64x48x28x28xf32) <- (64x48x28x28xf32, 1xf32)
        add_13 = paddle._C_ops.add(multiply_12, data_18)
        del data_18

        # builtin.combine: ([64x64x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32]) <- (64x64x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32)
        combine_2 = [batch_norm__60, add_11, add_12, add_13]

        # pd_op.concat: (64x208x28x28xf32) <- ([64x64x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32, 64x48x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (64x128x28x28xf32) <- (64x208x28x28xf32, 128x208x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_2, parameter_142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_142

        # pd_op.batch_norm_: (64x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (64x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_141,
                parameter_140,
                parameter_139,
                parameter_138,
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
        del parameter_138, parameter_139, parameter_140, parameter_141

        # pd_op.relu: (64x128x28x28xf32) <- (64x128x28x28xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.multiply: (64x128x28x28xf32) <- (1xf32, 64x128x28x28xf32)
        multiply_13 = paddle._C_ops.multiply(data_19, relu_13)
        del data_19

        # pd_op.add: (64x128x28x28xf32) <- (64x128x28x28xf32, 1xf32)
        add_14 = paddle._C_ops.add(multiply_13, data_20)
        del data_20

        # pd_op.conv2d: (64x256x28x28xf32) <- (64x128x28x28xf32, 256x128x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            add_14, parameter_137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_137

        # pd_op.batch_norm_: (64x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
                parameter_136,
                parameter_135,
                parameter_134,
                parameter_133,
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
        del parameter_133, parameter_134, parameter_135, parameter_136

        # pd_op.relu: (64x256x28x28xf32) <- (64x256x28x28xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.multiply: (64x256x28x28xf32) <- (1xf32, 64x256x28x28xf32)
        multiply_14 = paddle._C_ops.multiply(data_21, relu_14)
        del data_21

        # pd_op.add: (64x256x28x28xf32) <- (64x256x28x28xf32, 1xf32)
        add_15 = paddle._C_ops.add(multiply_14, data_22)
        del data_22

        # pd_op.depthwise_conv2d: (64x256x14x14xf32) <- (64x256x28x28xf32, 256x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            add_15, parameter_132, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_132

        # pd_op.batch_norm_: (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_1,
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

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x256x14x14xf32, 96x256x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_127, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_127

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_126,
                parameter_125,
                parameter_124,
                parameter_123,
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
        del parameter_123, parameter_124, parameter_125, parameter_126

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            batch_norm__102,
            parameter_122,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_122

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_121,
                parameter_120,
                parameter_119,
                parameter_118,
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
        del parameter_118, parameter_119, parameter_120, parameter_121

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_15 = paddle._C_ops.multiply(data_24, relu_15)
        del data_24

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_16 = paddle._C_ops.add(multiply_15, data_25)
        del data_25

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            add_16, parameter_117, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_117

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_116,
                parameter_115,
                parameter_114,
                parameter_113,
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
        del parameter_113, parameter_114, parameter_115, parameter_116

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            batch_norm__114,
            parameter_112,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_112

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_3,
                parameter_111,
                parameter_110,
                parameter_109,
                parameter_108,
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
        del parameter_108, parameter_109, parameter_110, parameter_111

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_16 = paddle._C_ops.multiply(data_26, relu_16)
        del data_26

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_17 = paddle._C_ops.add(multiply_16, data_27)
        del data_27

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            add_17, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_107

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_106,
                parameter_105,
                parameter_104,
                parameter_103,
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
        del parameter_103, parameter_104, parameter_105, parameter_106

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            batch_norm__126,
            parameter_102,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_102

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_4,
                parameter_101,
                parameter_100,
                parameter_99,
                parameter_98,
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
        del parameter_100, parameter_101, parameter_98, parameter_99

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_17 = paddle._C_ops.multiply(data_28, relu_17)
        del data_28

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_18 = paddle._C_ops.add(multiply_17, data_29)
        del data_29

        # builtin.combine: ([64x256x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32]) <- (64x256x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32)
        combine_3 = [batch_norm__96, add_16, add_17, add_18]

        # pd_op.concat: (64x544x14x14xf32) <- ([64x256x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (64x256x14x14xf32) <- (64x544x14x14xf32, 256x544x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            concat_3, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_97

        # pd_op.batch_norm_: (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
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

        # pd_op.relu: (64x256x14x14xf32) <- (64x256x14x14xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.multiply: (64x256x14x14xf32) <- (1xf32, 64x256x14x14xf32)
        multiply_18 = paddle._C_ops.multiply(data_30, relu_18)
        del data_30

        # pd_op.add: (64x256x14x14xf32) <- (64x256x14x14xf32, 1xf32)
        add_19 = paddle._C_ops.add(multiply_18, data_31)
        del data_31

        # pd_op.conv2d: (64x512x14x14xf32) <- (64x256x14x14xf32, 512x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            add_19, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_91,
                parameter_90,
                parameter_89,
                parameter_88,
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
        del parameter_88, parameter_89, parameter_90, parameter_91

        # pd_op.relu: (64x512x14x14xf32) <- (64x512x14x14xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.multiply: (64x512x14x14xf32) <- (1xf32, 64x512x14x14xf32)
        multiply_19 = paddle._C_ops.multiply(data_32, relu_19)
        del data_32

        # pd_op.add: (64x512x14x14xf32) <- (64x512x14x14xf32, 1xf32)
        add_20 = paddle._C_ops.add(multiply_19, data_33)
        del data_33

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x512x14x14xf32, 96x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            add_20, parameter_87, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_87

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_20,
                parameter_86,
                parameter_85,
                parameter_84,
                parameter_83,
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
        del parameter_83, parameter_84, parameter_85, parameter_86

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            batch_norm__150,
            parameter_82,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_82

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_5,
                parameter_81,
                parameter_80,
                parameter_79,
                parameter_78,
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
        del parameter_78, parameter_79, parameter_80, parameter_81

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_20 = paddle._C_ops.multiply(data_35, relu_20)
        del data_35

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_21 = paddle._C_ops.add(multiply_20, data_36)
        del data_36

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            add_21, parameter_77, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_76,
                parameter_75,
                parameter_74,
                parameter_73,
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
        del parameter_73, parameter_74, parameter_75, parameter_76

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            batch_norm__162,
            parameter_72,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_72

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_6,
                parameter_71,
                parameter_70,
                parameter_69,
                parameter_68,
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
        del parameter_68, parameter_69, parameter_70, parameter_71

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_21 = paddle._C_ops.multiply(data_37, relu_21)
        del data_37

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_22 = paddle._C_ops.add(multiply_21, data_38)
        del data_38

        # pd_op.conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x96x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            add_22, parameter_67, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_67

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_66,
                parameter_65,
                parameter_64,
                parameter_63,
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
        del parameter_63, parameter_64, parameter_65, parameter_66

        # pd_op.depthwise_conv2d: (64x96x14x14xf32) <- (64x96x14x14xf32, 96x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            batch_norm__174,
            parameter_62,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_62

        # pd_op.batch_norm_: (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (64x96x14x14xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_7,
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

        # pd_op.relu: (64x96x14x14xf32) <- (64x96x14x14xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.multiply: (64x96x14x14xf32) <- (1xf32, 64x96x14x14xf32)
        multiply_22 = paddle._C_ops.multiply(data_39, relu_22)
        del data_39

        # pd_op.add: (64x96x14x14xf32) <- (64x96x14x14xf32, 1xf32)
        add_23 = paddle._C_ops.add(multiply_22, data_40)
        del data_40

        # builtin.combine: ([64x512x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32]) <- (64x512x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32)
        combine_4 = [add_20, add_21, add_22, add_23]

        # pd_op.concat: (64x800x14x14xf32) <- ([64x512x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32, 64x96x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (64x256x14x14xf32) <- (64x800x14x14xf32, 256x800x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            concat_4, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.batch_norm_: (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (64x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
                parameter_56,
                parameter_55,
                parameter_54,
                parameter_53,
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
        del parameter_53, parameter_54, parameter_55, parameter_56

        # pd_op.relu: (64x256x14x14xf32) <- (64x256x14x14xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.multiply: (64x256x14x14xf32) <- (1xf32, 64x256x14x14xf32)
        multiply_23 = paddle._C_ops.multiply(data_41, relu_23)
        del data_41

        # pd_op.add: (64x256x14x14xf32) <- (64x256x14x14xf32, 1xf32)
        add_24 = paddle._C_ops.add(multiply_23, data_42)
        del data_42

        # pd_op.conv2d: (64x512x14x14xf32) <- (64x256x14x14xf32, 512x256x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            add_24, parameter_52, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_52

        # pd_op.batch_norm_: (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_51,
                parameter_50,
                parameter_49,
                parameter_48,
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
        del parameter_48, parameter_49, parameter_50, parameter_51

        # pd_op.relu: (64x512x14x14xf32) <- (64x512x14x14xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__192)
        del batch_norm__192

        # pd_op.multiply: (64x512x14x14xf32) <- (1xf32, 64x512x14x14xf32)
        multiply_24 = paddle._C_ops.multiply(data_43, relu_24)
        del data_43

        # pd_op.add: (64x512x14x14xf32) <- (64x512x14x14xf32, 1xf32)
        add_25 = paddle._C_ops.add(multiply_24, data_44)
        del data_44

        # pd_op.add: (64x512x14x14xf32) <- (64x512x14x14xf32, 64x512x14x14xf32)
        add_26 = paddle._C_ops.add(add_25, add_20)

        # pd_op.depthwise_conv2d: (64x512x7x7xf32) <- (64x512x14x14xf32, 512x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            add_26, parameter_47, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_47

        # pd_op.batch_norm_: (64x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_8,
                parameter_46,
                parameter_45,
                parameter_44,
                parameter_43,
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
        del parameter_43, parameter_44, parameter_45, parameter_46

        # pd_op.conv2d: (64x192x7x7xf32) <- (64x512x7x7xf32, 192x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_42, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_41,
                parameter_40,
                parameter_39,
                parameter_38,
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
        del parameter_38, parameter_39, parameter_40, parameter_41

        # pd_op.depthwise_conv2d: (64x192x7x7xf32) <- (64x192x7x7xf32, 192x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__204,
            parameter_37,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_37

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_9,
                parameter_36,
                parameter_35,
                parameter_34,
                parameter_33,
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
        del parameter_33, parameter_34, parameter_35, parameter_36

        # pd_op.relu: (64x192x7x7xf32) <- (64x192x7x7xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.multiply: (64x192x7x7xf32) <- (1xf32, 64x192x7x7xf32)
        multiply_25 = paddle._C_ops.multiply(data_46, relu_25)
        del data_46

        # pd_op.add: (64x192x7x7xf32) <- (64x192x7x7xf32, 1xf32)
        add_27 = paddle._C_ops.add(multiply_25, data_47)
        del data_47

        # pd_op.conv2d: (64x192x7x7xf32) <- (64x192x7x7xf32, 192x192x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            add_27, parameter_32, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_32

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_31,
                parameter_30,
                parameter_29,
                parameter_28,
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
        del parameter_28, parameter_29, parameter_30, parameter_31

        # pd_op.depthwise_conv2d: (64x192x7x7xf32) <- (64x192x7x7xf32, 192x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            batch_norm__216,
            parameter_27,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_27

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__222,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_10,
                parameter_26,
                parameter_25,
                parameter_24,
                parameter_23,
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
        del parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.relu: (64x192x7x7xf32) <- (64x192x7x7xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__222)
        del batch_norm__222

        # pd_op.multiply: (64x192x7x7xf32) <- (1xf32, 64x192x7x7xf32)
        multiply_26 = paddle._C_ops.multiply(data_48, relu_26)
        del data_48

        # pd_op.add: (64x192x7x7xf32) <- (64x192x7x7xf32, 1xf32)
        add_28 = paddle._C_ops.add(multiply_26, data_49)
        del data_49

        # pd_op.conv2d: (64x192x7x7xf32) <- (64x192x7x7xf32, 192x192x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            add_28, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__228,
            batch_norm__229,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
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

        # pd_op.depthwise_conv2d: (64x192x7x7xf32) <- (64x192x7x7xf32, 192x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            batch_norm__228,
            parameter_17,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_17

        # pd_op.batch_norm_: (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (64x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__234,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_11,
                parameter_16,
                parameter_15,
                parameter_14,
                parameter_13,
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
        del parameter_13, parameter_14, parameter_15, parameter_16

        # pd_op.relu: (64x192x7x7xf32) <- (64x192x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__234)
        del batch_norm__234

        # pd_op.multiply: (64x192x7x7xf32) <- (1xf32, 64x192x7x7xf32)
        multiply_27 = paddle._C_ops.multiply(data_50, relu_27)
        del data_50

        # pd_op.add: (64x192x7x7xf32) <- (64x192x7x7xf32, 1xf32)
        add_29 = paddle._C_ops.add(multiply_27, data_51)
        del data_51

        # builtin.combine: ([64x512x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32]) <- (64x512x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32)
        combine_5 = [batch_norm__198, add_27, add_28, add_29]

        # pd_op.concat: (64x1088x7x7xf32) <- ([64x512x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32, 64x192x7x7xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (64x512x7x7xf32) <- (64x1088x7x7xf32, 512x1088x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_5, parameter_12, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_12

        # pd_op.batch_norm_: (64x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (64x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__240,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_11,
                parameter_10,
                parameter_9,
                parameter_8,
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
        del parameter_10, parameter_11, parameter_8, parameter_9

        # pd_op.relu: (64x512x7x7xf32) <- (64x512x7x7xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__240)
        del batch_norm__240

        # pd_op.multiply: (64x512x7x7xf32) <- (1xf32, 64x512x7x7xf32)
        multiply_28 = paddle._C_ops.multiply(data_52, relu_28)
        del data_52

        # pd_op.add: (64x512x7x7xf32) <- (64x512x7x7xf32, 1xf32)
        add_30 = paddle._C_ops.add(multiply_28, data_53)
        del data_53

        # pd_op.conv2d: (64x1024x7x7xf32) <- (64x512x7x7xf32, 1024x512x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            add_30, parameter_7, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_7

        # pd_op.batch_norm_: (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (64x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_6,
                parameter_5,
                parameter_4,
                parameter_3,
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
        del parameter_3, parameter_4, parameter_5, parameter_6

        # pd_op.relu: (64x1024x7x7xf32) <- (64x1024x7x7xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__246)
        del batch_norm__246

        # pd_op.multiply: (64x1024x7x7xf32) <- (1xf32, 64x1024x7x7xf32)
        multiply_29 = paddle._C_ops.multiply(data_54, relu_29)
        del data_54

        # pd_op.add: (64x1024x7x7xf32) <- (64x1024x7x7xf32, 1xf32)
        add_31 = paddle._C_ops.add(multiply_29, data_55)
        del data_55

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (64x1024x1x1xf32) <- (64x1024x7x7xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            add_31,
            full_int_array_1,
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

        # pd_op.conv2d: (64x2048x1x1xf32) <- (64x1024x1x1xf32, 2048x1024x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            pool2d_1, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2

        # pd_op.relu: (64x2048x1x1xf32) <- (64x2048x1x1xf32)
        relu_30 = paddle._C_ops.relu(conv2d_30)
        del conv2d_30

        # pd_op.multiply: (64x2048x1x1xf32) <- (1xf32, 64x2048x1x1xf32)
        multiply_30 = paddle._C_ops.multiply(data_57, relu_30)
        del data_57

        # pd_op.add: (64x2048x1x1xf32) <- (64x2048x1x1xf32, 1xf32)
        add_32 = paddle._C_ops.add(multiply_30, data_58)
        del data_58

        # pd_op.flatten: (64x2048xf32) <- (64x2048x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(add_32, 1, 3)

        # pd_op.matmul: (64x102xf32) <- (64x2048xf32, 2048x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (64x102xf32) <- (64x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_14,
            add_15,
            add_16,
            add_17,
            add_18,
            add_19,
            add_2,
            add_20,
            add_21,
            add_22,
            add_23,
            add_24,
            add_25,
            add_26,
            add_27,
            add_28,
            add_29,
            add_3,
            add_30,
            add_31,
            add_32,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
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
            batch_norm__150,
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
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
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
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__22,
            batch_norm__220,
            batch_norm__221,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
            batch_norm__23,
            batch_norm__230,
            batch_norm__231,
            batch_norm__232,
            batch_norm__233,
            batch_norm__235,
            batch_norm__236,
            batch_norm__237,
            batch_norm__238,
            batch_norm__239,
            batch_norm__241,
            batch_norm__242,
            batch_norm__243,
            batch_norm__244,
            batch_norm__245,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__25,
            batch_norm__250,
            batch_norm__251,
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
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
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
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            concat_0,
            concat_1,
            concat_2,
            concat_3,
            concat_4,
            concat_5,
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
            conv2d_4,
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            depthwise_conv2d_0,
            depthwise_conv2d_1,
            depthwise_conv2d_10,
            depthwise_conv2d_11,
            depthwise_conv2d_2,
            depthwise_conv2d_3,
            depthwise_conv2d_4,
            depthwise_conv2d_5,
            depthwise_conv2d_6,
            depthwise_conv2d_7,
            depthwise_conv2d_8,
            depthwise_conv2d_9,
            flatten_0,
            full_0,
            full_int_array_0,
            full_int_array_1,
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
            multiply_27,
            multiply_28,
            multiply_29,
            multiply_3,
            multiply_30,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            parameter_0,
            pool2d_0,
            pool2d_1,
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
            relu_4,
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
        )

        return add_0
