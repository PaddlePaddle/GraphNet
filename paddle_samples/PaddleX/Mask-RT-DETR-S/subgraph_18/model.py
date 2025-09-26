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
        data_0,
    ):
        # pd_op.conv2d: (1x24x-1x-1xf32) <- (1x3x-1x-1xf32, 24x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_209, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_209

        # pd_op.batch_norm_: (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_208,
                parameter_207,
                parameter_206,
                parameter_205,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_0, parameter_205, parameter_206, parameter_207, parameter_208

        # pd_op.relu: (1x24x-1x-1xf32) <- (1x24x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (1x12x-1x-1xf32) <- (1x24x-1x-1xf32, 12x24x2x2xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_1, parameter_204, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_204

        # pd_op.batch_norm_: (1x12x-1x-1xf32, 12xf32, 12xf32, 12xf32, 12xf32, -1xui8) <- (1x12x-1x-1xf32, 12xf32, 12xf32, 12xf32, 12xf32)
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
                parameter_203,
                parameter_202,
                parameter_201,
                parameter_200,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_1, parameter_200, parameter_201, parameter_202, parameter_203

        # pd_op.relu: (1x12x-1x-1xf32) <- (1x12x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (1x24x-1x-1xf32) <- (1x12x-1x-1xf32, 24x12x2x2xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_2, parameter_199, [1, 1], [0, 0], "SAME", [1, 1], 1, "NCHW"
        )
        del parameter_199, relu_2

        # pd_op.batch_norm_: (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_198,
                parameter_197,
                parameter_196,
                parameter_195,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_2, parameter_195, parameter_196, parameter_197, parameter_198

        # pd_op.relu: (1x24x-1x-1xf32) <- (1x24x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 2]

        # pd_op.pool2d: (1x24x-1x-1xf32) <- (1x24x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_1,
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
        del full_int_array_0, relu_1

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

        # builtin.combine: ([1x24x-1x-1xf32, 1x24x-1x-1xf32]) <- (1x24x-1x-1xf32, 1x24x-1x-1xf32)
        combine_0 = [pool2d_0, relu_3]
        del pool2d_0, relu_3

        # pd_op.concat: (1x48x-1x-1xf32) <- ([1x24x-1x-1xf32, 1x24x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (1x24x-1x-1xf32) <- (1x48x-1x-1xf32, 24x48x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            concat_0, parameter_194, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_194

        # pd_op.batch_norm_: (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (1x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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
                parameter_193,
                parameter_192,
                parameter_191,
                parameter_190,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_3, parameter_190, parameter_191, parameter_192, parameter_193

        # pd_op.relu: (1x24x-1x-1xf32) <- (1x24x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x24x-1x-1xf32, 32x24x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_4, parameter_189, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_189, relu_4

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_188,
                parameter_187,
                parameter_186,
                parameter_185,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_4, parameter_185, parameter_186, parameter_187, parameter_188

        # pd_op.relu: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_5, parameter_184, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_184

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_183,
                parameter_182,
                parameter_181,
                parameter_180,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_180, parameter_181, parameter_182, parameter_183

        # pd_op.relu: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_6, parameter_179, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_179

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_178,
                parameter_177,
                parameter_176,
                parameter_175,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_175, parameter_176, parameter_177, parameter_178

        # pd_op.relu: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32, 32x32x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_7, parameter_174, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_174

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_173,
                parameter_172,
                parameter_171,
                parameter_170,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_170, parameter_171, parameter_172, parameter_173

        # pd_op.relu: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # builtin.combine: ([1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32]) <- (1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32)
        combine_1 = [relu_5, relu_6, relu_7, relu_8]

        # pd_op.concat: (1x128x-1x-1xf32) <- ([1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32, 1x32x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (1x32x-1x-1xf32) <- (1x128x-1x-1xf32, 32x128x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_1, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_169

        # pd_op.batch_norm_: (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (1x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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
                parameter_168,
                parameter_167,
                parameter_166,
                parameter_165,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_165, parameter_166, parameter_167, parameter_168

        # pd_op.relu: (1x32x-1x-1xf32) <- (1x32x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (1x64x-1x-1xf32) <- (1x32x-1x-1xf32, 64x32x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            relu_9, parameter_164, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_164

        # pd_op.batch_norm_: (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_163,
                parameter_162,
                parameter_161,
                parameter_160,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_160, parameter_161, parameter_162, parameter_163

        # pd_op.relu: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.depthwise_conv2d: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 64x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            relu_10, parameter_159, [2, 2], [1, 1], "EXPLICIT", 64, [1, 1], "NCHW"
        )
        del parameter_159

        # pd_op.batch_norm_: (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
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
                parameter_158,
                parameter_157,
                parameter_156,
                parameter_155,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_155, parameter_156, parameter_157, parameter_158

        # pd_op.conv2d: (1x48x-1x-1xf32) <- (1x64x-1x-1xf32, 48x64x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__60, parameter_154, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_154

        # pd_op.batch_norm_: (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_153,
                parameter_152,
                parameter_151,
                parameter_150,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_150, parameter_151, parameter_152, parameter_153

        # pd_op.relu: (1x48x-1x-1xf32) <- (1x48x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (1x48x-1x-1xf32) <- (1x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_11, parameter_149, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_149

        # pd_op.batch_norm_: (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_148,
                parameter_147,
                parameter_146,
                parameter_145,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_145, parameter_146, parameter_147, parameter_148

        # pd_op.relu: (1x48x-1x-1xf32) <- (1x48x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (1x48x-1x-1xf32) <- (1x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_12, parameter_144, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_144

        # pd_op.batch_norm_: (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (1x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                parameter_143,
                parameter_142,
                parameter_141,
                parameter_140,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_140, parameter_141, parameter_142, parameter_143

        # pd_op.relu: (1x48x-1x-1xf32) <- (1x48x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # builtin.combine: ([1x64x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32]) <- (1x64x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32)
        combine_2 = [batch_norm__60, relu_11, relu_12, relu_13]

        # pd_op.concat: (1x208x-1x-1xf32) <- ([1x64x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32, 1x48x-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (1x128x-1x-1xf32) <- (1x208x-1x-1xf32, 128x208x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_2, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_139

        # pd_op.batch_norm_: (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (1x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_138,
                parameter_137,
                parameter_136,
                parameter_135,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_135, parameter_136, parameter_137, parameter_138

        # pd_op.relu: (1x128x-1x-1xf32) <- (1x128x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x128x-1x-1xf32, 256x128x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_14, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_134

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_133,
                parameter_132,
                parameter_131,
                parameter_130,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_130, parameter_131, parameter_132, parameter_133

        # pd_op.relu: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.depthwise_conv2d: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32, 256x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            relu_15, parameter_129, [2, 2], [1, 1], "EXPLICIT", 256, [1, 1], "NCHW"
        )
        del parameter_129

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_128,
                parameter_127,
                parameter_126,
                parameter_125,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_125, parameter_126, parameter_127, parameter_128

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x256x-1x-1xf32, 96x256x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_124, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_124

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_123,
                parameter_122,
                parameter_121,
                parameter_120,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_120, parameter_121, parameter_122, parameter_123

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            batch_norm__102,
            parameter_119,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_119

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_118,
                parameter_117,
                parameter_116,
                parameter_115,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_115, parameter_116, parameter_117, parameter_118

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_16, parameter_114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_114

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_113,
                parameter_112,
                parameter_111,
                parameter_110,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_110, parameter_111, parameter_112, parameter_113

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(
            batch_norm__114,
            parameter_109,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_109

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_108,
                parameter_107,
                parameter_106,
                parameter_105,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_105, parameter_106, parameter_107, parameter_108

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_17, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_104

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_100, parameter_101, parameter_102, parameter_103

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(
            batch_norm__126,
            parameter_99,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_99

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_98,
                parameter_97,
                parameter_96,
                parameter_95,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_95, parameter_96, parameter_97, parameter_98

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # builtin.combine: ([1x256x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32]) <- (1x256x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32)
        combine_3 = [batch_norm__96, relu_16, relu_17, relu_18]

        # pd_op.concat: (1x544x-1x-1xf32) <- ([1x256x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x544x-1x-1xf32, 256x544x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            concat_3, parameter_94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_94

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_93,
                parameter_92,
                parameter_91,
                parameter_90,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_90, parameter_91, parameter_92, parameter_93

        # pd_op.relu: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_19, parameter_89, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_88,
                parameter_87,
                parameter_86,
                parameter_85,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_85, parameter_86, parameter_87, parameter_88

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x512x-1x-1xf32, 96x512x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_20, parameter_84, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_83,
                parameter_82,
                parameter_81,
                parameter_80,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_80, parameter_81, parameter_82, parameter_83

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(
            batch_norm__150,
            parameter_79,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_79

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_78,
                parameter_77,
                parameter_76,
                parameter_75,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_75, parameter_76, parameter_77, parameter_78

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_21, parameter_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_74

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_73,
                parameter_72,
                parameter_71,
                parameter_70,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_70, parameter_71, parameter_72, parameter_73

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(
            batch_norm__162,
            parameter_69,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_69

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            relu_22, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_63,
                parameter_62,
                parameter_61,
                parameter_60,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_60, parameter_61, parameter_62, parameter_63

        # pd_op.depthwise_conv2d: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32, 96x1x5x5xf32)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(
            batch_norm__174,
            parameter_59,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            96,
            [1, 1],
            "NCHW",
        )
        del parameter_59

        # pd_op.batch_norm_: (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (1x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                parameter_58,
                parameter_57,
                parameter_56,
                parameter_55,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_55, parameter_56, parameter_57, parameter_58

        # pd_op.relu: (1x96x-1x-1xf32) <- (1x96x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # builtin.combine: ([1x512x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32]) <- (1x512x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32)
        combine_4 = [relu_20, relu_21, relu_22, relu_23]

        # pd_op.concat: (1x800x-1x-1xf32) <- ([1x512x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32, 1x96x-1x-1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (1x256x-1x-1xf32) <- (1x800x-1x-1xf32, 256x800x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            concat_4, parameter_54, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_54

        # pd_op.batch_norm_: (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_53,
                parameter_52,
                parameter_51,
                parameter_50,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_50, parameter_51, parameter_52, parameter_53

        # pd_op.relu: (1x256x-1x-1xf32) <- (1x256x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x256x-1x-1xf32, 512x256x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            relu_24, parameter_49, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_48,
                parameter_47,
                parameter_46,
                parameter_45,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_45, parameter_46, parameter_47, parameter_48

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__192)
        del batch_norm__192

        # pd_op.add: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 1x512x-1x-1xf32)
        add_0 = paddle._C_ops.add(relu_25, relu_20)

        # pd_op.depthwise_conv2d: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32, 512x1x3x3xf32)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(
            add_0, parameter_44, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del parameter_44

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_43,
                parameter_42,
                parameter_41,
                parameter_40,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_40, parameter_41, parameter_42, parameter_43

        # pd_op.conv2d: (1x192x-1x-1xf32) <- (1x512x-1x-1xf32, 192x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            batch_norm__198, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_39

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.depthwise_conv2d: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32, 192x1x5x5xf32)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(
            batch_norm__204,
            parameter_34,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_34

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.conv2d: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_26, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.depthwise_conv2d: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32, 192x1x5x5xf32)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(
            batch_norm__216,
            parameter_24,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_24

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.relu: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__222)
        del batch_norm__222

        # pd_op.conv2d: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_27, parameter_19, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.depthwise_conv2d: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32, 192x1x5x5xf32)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(
            batch_norm__228,
            parameter_14,
            [1, 1],
            [2, 2],
            "EXPLICIT",
            192,
            [1, 1],
            "NCHW",
        )
        del parameter_14

        # pd_op.batch_norm_: (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (1x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.relu: (1x192x-1x-1xf32) <- (1x192x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__234)
        del batch_norm__234

        # builtin.combine: ([1x512x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32]) <- (1x512x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32)
        combine_5 = [batch_norm__198, relu_26, relu_27, relu_28]

        # pd_op.concat: (1x1088x-1x-1xf32) <- ([1x512x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32, 1x192x-1x-1xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5, full_0

        # pd_op.conv2d: (1x512x-1x-1xf32) <- (1x1088x-1x-1xf32, 512x1088x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_5, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # pd_op.batch_norm_: (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (1x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.relu: (1x512x-1x-1xf32) <- (1x512x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__240)
        del batch_norm__240

        # pd_op.conv2d: (1x1024x-1x-1xf32) <- (1x512x-1x-1xf32, 1024x512x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_29, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

        # pd_op.batch_norm_: (1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (1x1024x-1x-1xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_3,
                parameter_2,
                parameter_1,
                parameter_0,
                False,
                float("0.9"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del parameter_0, parameter_1, parameter_2, parameter_3

        # pd_op.relu: (1x1024x-1x-1xf32) <- (1x1024x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__246)
        del (
            add_0,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__109,
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
            batch_norm__130,
            batch_norm__131,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            batch_norm__169,
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
            batch_norm__190,
            batch_norm__191,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            batch_norm__198,
            batch_norm__199,
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
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
            batch_norm__223,
            batch_norm__224,
            batch_norm__225,
            batch_norm__226,
            batch_norm__227,
            batch_norm__228,
            batch_norm__229,
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
            batch_norm__246,
            batch_norm__247,
            batch_norm__248,
            batch_norm__249,
            batch_norm__250,
            batch_norm__251,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__49,
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
            batch_norm__70,
            batch_norm__71,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
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
            concat_2,
            concat_3,
            concat_4,
            concat_5,
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
            relu_5,
            relu_6,
            relu_7,
            relu_8,
            relu_9,
        )

        return relu_0
