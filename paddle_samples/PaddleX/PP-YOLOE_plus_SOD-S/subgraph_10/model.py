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
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
    ):
        # pd_op.conv2d: (2x16x-1x-1xf32) <- (2x3x-1x-1xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_6, parameter_192, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_6, parameter_192

        # pd_op.batch_norm_: (2x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (2x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.swish: (2x16x-1x-1xf32) <- (2x16x-1x-1xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (2x16x-1x-1xf32) <- (2x16x-1x-1xf32, 16x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            swish_1, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187

        # pd_op.batch_norm_: (2x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32, -1xui8) <- (2x16x-1x-1xf32, 16xf32, 16xf32, 16xf32, 16xf32)
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

        # pd_op.swish: (2x16x-1x-1xf32) <- (2x16x-1x-1xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (2x32x-1x-1xf32) <- (2x16x-1x-1xf32, 32x16x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            swish_2, parameter_182, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182

        # pd_op.batch_norm_: (2x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32, -1xui8) <- (2x32x-1x-1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
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

        # pd_op.swish: (2x32x-1x-1xf32) <- (2x32x-1x-1xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x32x-1x-1xf32, 48x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            swish_3, parameter_177, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_177

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.conv2d: (2x24x-1x-1xf32) <- (2x48x-1x-1xf32, 24x48x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_4, parameter_172, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_172

        # pd_op.batch_norm_: (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.swish: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.conv2d: (2x24x-1x-1xf32) <- (2x48x-1x-1xf32, 24x48x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            swish_4, parameter_167, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167

        # pd_op.batch_norm_: (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.swish: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__30)

        # pd_op.conv2d: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32, 24x24x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            swish_6, parameter_162, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_162

        # pd_op.batch_norm_: (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.swish: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.conv2d: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32, 24x24x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            swish_7, parameter_157, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_157

        # pd_op.batch_norm_: (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.conv2d: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32, 24x24x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            swish_7, parameter_152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_152

        # pd_op.batch_norm_: (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32, -1xui8) <- (2x24x-1x-1xf32, 24xf32, 24xf32, 24xf32, 24xf32)
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

        # pd_op.multiply: (2x24x-1x-1xf32) <- (1xf32, 2x24x-1x-1xf32)
        multiply_0 = paddle._C_ops.multiply(data_0, batch_norm__48)
        del data_0

        # pd_op.add: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32, 2x24x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__42, multiply_0)

        # pd_op.swish: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32)
        swish_8 = paddle._C_ops.swish(add_0)

        # pd_op.add: (2x24x-1x-1xf32) <- (2x24x-1x-1xf32, 2x24x-1x-1xf32)
        add_1 = paddle._C_ops.add(swish_6, swish_8)

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

        # builtin.combine: ([2x24x-1x-1xf32, 2x24x-1x-1xf32]) <- (2x24x-1x-1xf32, 2x24x-1x-1xf32)
        combine_0 = [swish_5, add_1]

        # pd_op.concat: (2x48x-1x-1xf32) <- ([2x24x-1x-1xf32, 2x24x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 3]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.mean: (2x48x1x1xf32) <- (2x48x-1x-1xf32, 2xi64)
        mean_0 = paddle._C_ops.mean(concat_0, full_int_array_0, True)

        # pd_op.conv2d: (2x48x1x1xf32) <- (2x48x1x1xf32, 48x48x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            mean_0, parameter_147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32) <- (48xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_146, full_int_array_1)
        del parameter_146

        # pd_op.add: (2x48x1x1xf32) <- (2x48x1x1xf32, 1x48x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_9, reshape_0)

        # pd_op.hardsigmoid: (2x48x1x1xf32) <- (2x48x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(
            add_2, float("0.166667"), float("0.5")
        )
        del add_2

        # pd_op.multiply: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 2x48x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(concat_0, hardsigmoid_0)

        # pd_op.conv2d: (2x64x-1x-1xf32) <- (2x48x-1x-1xf32, 64x48x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            multiply_1, parameter_145, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_145

        # pd_op.batch_norm_: (2x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (2x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_10,
                parameter_144,
                parameter_143,
                parameter_142,
                parameter_141,
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
        del parameter_141, parameter_142, parameter_143, parameter_144

        # pd_op.swish: (2x64x-1x-1xf32) <- (2x64x-1x-1xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x64x-1x-1xf32, 96x64x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            swish_9, parameter_140, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_140

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_139,
                parameter_138,
                parameter_137,
                parameter_136,
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
        del parameter_136, parameter_137, parameter_138, parameter_139

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x96x-1x-1xf32, 48x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            swish_10, parameter_135, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_135

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_12,
                parameter_134,
                parameter_133,
                parameter_132,
                parameter_131,
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
        del parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_11 = paddle._C_ops.swish(batch_norm__66)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x96x-1x-1xf32, 48x96x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            swish_10, parameter_130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_130

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_129,
                parameter_128,
                parameter_127,
                parameter_126,
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
        del parameter_126, parameter_127, parameter_128, parameter_129

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__72)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            swish_12, parameter_125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_125

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_14,
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

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            swish_13, parameter_120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_15,
                parameter_119,
                parameter_118,
                parameter_117,
                parameter_116,
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
        del parameter_116, parameter_117, parameter_118, parameter_119

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            swish_13, parameter_115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_115

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_114,
                parameter_113,
                parameter_112,
                parameter_111,
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
        del parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.multiply: (2x48x-1x-1xf32) <- (1xf32, 2x48x-1x-1xf32)
        multiply_2 = paddle._C_ops.multiply(data_1, batch_norm__90)
        del data_1

        # pd_op.add: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 2x48x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__84, multiply_2)

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_14 = paddle._C_ops.swish(add_3)

        # pd_op.add: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 2x48x-1x-1xf32)
        add_4 = paddle._C_ops.add(swish_12, swish_14)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            add_4, parameter_110, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_110

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_17,
                parameter_109,
                parameter_108,
                parameter_107,
                parameter_106,
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
        del parameter_106, parameter_107, parameter_108, parameter_109

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            swish_15, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_18,
                parameter_104,
                parameter_103,
                parameter_102,
                parameter_101,
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
        del parameter_101, parameter_102, parameter_103, parameter_104

        # pd_op.conv2d: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 48x48x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            swish_15, parameter_100, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_100

        # pd_op.batch_norm_: (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x-1x-1xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        (
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_99,
                parameter_98,
                parameter_97,
                parameter_96,
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
        del parameter_96, parameter_97, parameter_98, parameter_99

        # pd_op.multiply: (2x48x-1x-1xf32) <- (1xf32, 2x48x-1x-1xf32)
        multiply_3 = paddle._C_ops.multiply(data_2, batch_norm__108)
        del data_2

        # pd_op.add: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 2x48x-1x-1xf32)
        add_5 = paddle._C_ops.add(batch_norm__102, multiply_3)

        # pd_op.swish: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32)
        swish_16 = paddle._C_ops.swish(add_5)

        # pd_op.add: (2x48x-1x-1xf32) <- (2x48x-1x-1xf32, 2x48x-1x-1xf32)
        add_6 = paddle._C_ops.add(add_4, swish_16)

        # builtin.combine: ([2x48x-1x-1xf32, 2x48x-1x-1xf32]) <- (2x48x-1x-1xf32, 2x48x-1x-1xf32)
        combine_1 = [swish_11, add_6]

        # pd_op.concat: (2x96x-1x-1xf32) <- ([2x48x-1x-1xf32, 2x48x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.mean: (2x96x1x1xf32) <- (2x96x-1x-1xf32, 2xi64)
        mean_1 = paddle._C_ops.mean(concat_1, full_int_array_0, True)

        # pd_op.conv2d: (2x96x1x1xf32) <- (2x96x1x1xf32, 96x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            mean_1, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.reshape: (1x96x1x1xf32) <- (96xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_94, full_int_array_1)
        del parameter_94

        # pd_op.add: (2x96x1x1xf32) <- (2x96x1x1xf32, 1x96x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_20, reshape_1)

        # pd_op.hardsigmoid: (2x96x1x1xf32) <- (2x96x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(
            add_7, float("0.166667"), float("0.5")
        )
        del add_7

        # pd_op.multiply: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 2x96x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(concat_1, hardsigmoid_1)

        # pd_op.conv2d: (2x128x-1x-1xf32) <- (2x96x-1x-1xf32, 128x96x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            multiply_4, parameter_93, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_93

        # pd_op.batch_norm_: (2x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (2x128x-1x-1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_21,
                parameter_92,
                parameter_91,
                parameter_90,
                parameter_89,
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
        del parameter_89, parameter_90, parameter_91, parameter_92

        # pd_op.swish: (2x128x-1x-1xf32) <- (2x128x-1x-1xf32)
        swish_17 = paddle._C_ops.swish(batch_norm__114)

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x128x-1x-1xf32, 192x128x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            swish_17, parameter_88, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_87,
                parameter_86,
                parameter_85,
                parameter_84,
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
        del parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.swish: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__120)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x192x-1x-1xf32, 96x192x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            swish_18, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_23,
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

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__126)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x192x-1x-1xf32, 96x192x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            swish_18, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_77,
                parameter_76,
                parameter_75,
                parameter_74,
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
        del parameter_74, parameter_75, parameter_76, parameter_77

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_20 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            swish_20, parameter_73, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_73

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_72,
                parameter_71,
                parameter_70,
                parameter_69,
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
        del parameter_69, parameter_70, parameter_71, parameter_72

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_21 = paddle._C_ops.swish(batch_norm__138)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            swish_21, parameter_68, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_68

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_67,
                parameter_66,
                parameter_65,
                parameter_64,
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
        del parameter_64, parameter_65, parameter_66, parameter_67

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            swish_21, parameter_63, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_62,
                parameter_61,
                parameter_60,
                parameter_59,
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
        del parameter_59, parameter_60, parameter_61, parameter_62

        # pd_op.multiply: (2x96x-1x-1xf32) <- (1xf32, 2x96x-1x-1xf32)
        multiply_5 = paddle._C_ops.multiply(data_3, batch_norm__150)
        del data_3

        # pd_op.add: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 2x96x-1x-1xf32)
        add_8 = paddle._C_ops.add(batch_norm__144, multiply_5)

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_22 = paddle._C_ops.swish(add_8)

        # pd_op.add: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 2x96x-1x-1xf32)
        add_9 = paddle._C_ops.add(swish_20, swish_22)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            add_9, parameter_58, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_58

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_57,
                parameter_56,
                parameter_55,
                parameter_54,
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
        del parameter_54, parameter_55, parameter_56, parameter_57

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_23 = paddle._C_ops.swish(batch_norm__156)

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            swish_23, parameter_53, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_53

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_52,
                parameter_51,
                parameter_50,
                parameter_49,
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
        del parameter_49, parameter_50, parameter_51, parameter_52

        # pd_op.conv2d: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 96x96x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            swish_23, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_48

        # pd_op.batch_norm_: (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x-1x-1xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_30,
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

        # pd_op.multiply: (2x96x-1x-1xf32) <- (1xf32, 2x96x-1x-1xf32)
        multiply_6 = paddle._C_ops.multiply(data_4, batch_norm__168)
        del data_4

        # pd_op.add: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 2x96x-1x-1xf32)
        add_10 = paddle._C_ops.add(batch_norm__162, multiply_6)

        # pd_op.swish: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32)
        swish_24 = paddle._C_ops.swish(add_10)

        # pd_op.add: (2x96x-1x-1xf32) <- (2x96x-1x-1xf32, 2x96x-1x-1xf32)
        add_11 = paddle._C_ops.add(add_9, swish_24)

        # builtin.combine: ([2x96x-1x-1xf32, 2x96x-1x-1xf32]) <- (2x96x-1x-1xf32, 2x96x-1x-1xf32)
        combine_2 = [swish_19, add_11]

        # pd_op.concat: (2x192x-1x-1xf32) <- ([2x96x-1x-1xf32, 2x96x-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.mean: (2x192x1x1xf32) <- (2x192x-1x-1xf32, 2xi64)
        mean_2 = paddle._C_ops.mean(concat_2, full_int_array_0, True)

        # pd_op.conv2d: (2x192x1x1xf32) <- (2x192x1x1xf32, 192x192x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            mean_2, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43

        # pd_op.reshape: (1x192x1x1xf32) <- (192xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_42, full_int_array_1)
        del parameter_42

        # pd_op.add: (2x192x1x1xf32) <- (2x192x1x1xf32, 1x192x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_31, reshape_2)

        # pd_op.hardsigmoid: (2x192x1x1xf32) <- (2x192x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(
            add_12, float("0.166667"), float("0.5")
        )
        del add_12

        # pd_op.multiply: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 2x192x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(concat_2, hardsigmoid_2)

        # pd_op.conv2d: (2x256x-1x-1xf32) <- (2x192x-1x-1xf32, 256x192x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            multiply_7, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.batch_norm_: (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (2x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
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

        # pd_op.swish: (2x256x-1x-1xf32) <- (2x256x-1x-1xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.conv2d: (2x384x-1x-1xf32) <- (2x256x-1x-1xf32, 384x256x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            swish_25, parameter_36, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.batch_norm_: (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x-1x-1xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_35,
                parameter_34,
                parameter_33,
                parameter_32,
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
        del parameter_32, parameter_33, parameter_34, parameter_35

        # pd_op.swish: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32)
        swish_26 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x384x-1x-1xf32, 192x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            swish_26, parameter_31, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_31

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_30,
                parameter_29,
                parameter_28,
                parameter_27,
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
        del parameter_27, parameter_28, parameter_29, parameter_30

        # pd_op.swish: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x384x-1x-1xf32, 192x384x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            swish_26, parameter_26, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_26

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_25,
                parameter_24,
                parameter_23,
                parameter_22,
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
        del parameter_22, parameter_23, parameter_24, parameter_25

        # pd_op.swish: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__192)

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 192x192x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            swish_28, parameter_21, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.swish: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32)
        swish_29 = paddle._C_ops.swish(batch_norm__198)

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 192x192x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            swish_29, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
                parameter_15,
                parameter_14,
                parameter_13,
                parameter_12,
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
        del parameter_12, parameter_13, parameter_14, parameter_15

        # pd_op.conv2d: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 192x192x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            swish_29, parameter_11, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_11

        # pd_op.batch_norm_: (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x-1x-1xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
                parameter_10,
                parameter_9,
                parameter_8,
                parameter_7,
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
        del parameter_10, parameter_7, parameter_8, parameter_9

        # pd_op.multiply: (2x192x-1x-1xf32) <- (1xf32, 2x192x-1x-1xf32)
        multiply_8 = paddle._C_ops.multiply(data_5, batch_norm__210)
        del data_5

        # pd_op.add: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 2x192x-1x-1xf32)
        add_13 = paddle._C_ops.add(batch_norm__204, multiply_8)

        # pd_op.swish: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32)
        swish_30 = paddle._C_ops.swish(add_13)

        # pd_op.add: (2x192x-1x-1xf32) <- (2x192x-1x-1xf32, 2x192x-1x-1xf32)
        add_14 = paddle._C_ops.add(swish_28, swish_30)

        # builtin.combine: ([2x192x-1x-1xf32, 2x192x-1x-1xf32]) <- (2x192x-1x-1xf32, 2x192x-1x-1xf32)
        combine_3 = [swish_27, add_14]

        # pd_op.concat: (2x384x-1x-1xf32) <- ([2x192x-1x-1xf32, 2x192x-1x-1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.mean: (2x384x1x1xf32) <- (2x384x-1x-1xf32, 2xi64)
        mean_3 = paddle._C_ops.mean(concat_3, full_int_array_0, True)

        # pd_op.conv2d: (2x384x1x1xf32) <- (2x384x1x1xf32, 384x384x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            mean_3, parameter_6, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_6

        # pd_op.reshape: (1x384x1x1xf32) <- (384xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_5, full_int_array_1)
        del full_int_array_1, parameter_5

        # pd_op.add: (2x384x1x1xf32) <- (2x384x1x1xf32, 1x384x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_39, reshape_3)

        # pd_op.hardsigmoid: (2x384x1x1xf32) <- (2x384x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(
            add_15, float("0.166667"), float("0.5")
        )
        del add_15

        # pd_op.multiply: (2x384x-1x-1xf32) <- (2x384x-1x-1xf32, 2x384x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(concat_3, hardsigmoid_3)

        # pd_op.conv2d: (2x512x-1x-1xf32) <- (2x384x-1x-1xf32, 512x384x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            multiply_9, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4

        # pd_op.batch_norm_: (2x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (2x512x-1x-1xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__216,
            batch_norm__217,
            batch_norm__218,
            batch_norm__219,
            batch_norm__220,
            batch_norm__221,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
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

        # pd_op.swish: (2x512x-1x-1xf32) <- (2x512x-1x-1xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__216)
        del (
            add_0,
            add_1,
            add_10,
            add_11,
            add_13,
            add_14,
            add_3,
            add_4,
            add_5,
            add_6,
            add_8,
            add_9,
            assign_0,
            assign_1,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
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
            batch_norm__210,
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
            concat_0,
            concat_1,
            concat_2,
            concat_3,
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
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            full_0,
            full_int_array_0,
            hardsigmoid_0,
            hardsigmoid_1,
            hardsigmoid_2,
            hardsigmoid_3,
            mean_0,
            mean_1,
            mean_2,
            mean_3,
            multiply_0,
            multiply_1,
            multiply_2,
            multiply_3,
            multiply_4,
            multiply_5,
            multiply_6,
            multiply_7,
            multiply_8,
            multiply_9,
            reshape_0,
            reshape_1,
            reshape_2,
            reshape_3,
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
            swish_30,
            swish_4,
            swish_5,
            swish_6,
            swish_7,
            swish_8,
            swish_9,
        )

        return swish_0
