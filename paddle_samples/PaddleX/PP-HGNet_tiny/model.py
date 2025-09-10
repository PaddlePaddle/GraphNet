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
    ):
        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x3x224x224xf32, 48x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_192, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_192

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                True,
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
        del conv2d_0, parameter_188, parameter_189, parameter_190, parameter_191

        # pd_op.relu: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)
        del batch_norm__0

        # pd_op.conv2d: (-1x48x112x112xf32) <- (-1x48x112x112xf32, 48x48x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            relu_0, parameter_187, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_187, relu_0

        # pd_op.batch_norm_: (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (-1x48x112x112xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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
                True,
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
        del conv2d_1, parameter_183, parameter_184, parameter_185, parameter_186

        # pd_op.relu: (-1x48x112x112xf32) <- (-1x48x112x112xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x96x112x112xf32) <- (-1x48x112x112xf32, 96x48x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            relu_1, parameter_182, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_182, relu_1

        # pd_op.batch_norm_: (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x112x112xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_2, parameter_178, parameter_179, parameter_180, parameter_181

        # pd_op.relu: (-1x96x112x112xf32) <- (-1x96x112x112xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)
        del batch_norm__12

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x96x56x56xf32) <- (-1x96x112x112xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            relu_2,
            full_int_array_0,
            [2, 2],
            [1, 1],
            False,
            True,
            "NCHW",
            "max",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_0, relu_2

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            pool2d_0, parameter_177, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_177

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_3, parameter_173, parameter_174, parameter_175, parameter_176

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_3, parameter_172, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_172

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_4, parameter_168, parameter_169, parameter_170, parameter_171

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            relu_4, parameter_167, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_167

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_5, parameter_163, parameter_164, parameter_165, parameter_166

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__30)
        del batch_norm__30

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            relu_5, parameter_162, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_162

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_6, parameter_158, parameter_159, parameter_160, parameter_161

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x96x56x56xf32) <- (-1x96x56x56xf32, 96x96x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_6, parameter_157, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_157

        # pd_op.batch_norm_: (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (-1x96x56x56xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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
                True,
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
        del conv2d_7, parameter_153, parameter_154, parameter_155, parameter_156

        # pd_op.relu: (-1x96x56x56xf32) <- (-1x96x56x56xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32]) <- (-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32)
        combine_0 = [pool2d_0, relu_3, relu_4, relu_5, relu_6, relu_7]
        del pool2d_0, relu_3, relu_4, relu_5, relu_6, relu_7

        # pd_op.concat: (-1x576x56x56xf32) <- ([-1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32, -1x96x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0, full_0

        # pd_op.conv2d: (-1x224x56x56xf32) <- (-1x576x56x56xf32, 224x576x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_0, parameter_152, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_152

        # pd_op.batch_norm_: (-1x224x56x56xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x56x56xf32, 224xf32, 224xf32, 224xf32, 224xf32)
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
                True,
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
        del conv2d_8, parameter_148, parameter_149, parameter_150, parameter_151

        # pd_op.relu: (-1x224x56x56xf32) <- (-1x224x56x56xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x224x1x1xf32) <- (-1x224x56x56xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            relu_8,
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
        del full_int_array_1

        # pd_op.conv2d: (-1x224x1x1xf32) <- (-1x224x1x1xf32, 224x224x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            pool2d_1, parameter_147, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_147, pool2d_1

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, -1, 1, 1]

        # pd_op.reshape: (1x224x1x1xf32) <- (224xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_146, full_int_array_2)
        del full_int_array_2, parameter_146

        # pd_op.add: (-1x224x1x1xf32) <- (-1x224x1x1xf32, 1x224x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_9, reshape_0)
        del conv2d_9, reshape_0

        # pd_op.sigmoid: (-1x224x1x1xf32) <- (-1x224x1x1xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(add_1)
        del add_1

        # pd_op.multiply: (-1x224x56x56xf32) <- (-1x224x56x56xf32, -1x224x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(relu_8, sigmoid_0)
        del relu_8, sigmoid_0

        # pd_op.depthwise_conv2d: (-1x224x28x28xf32) <- (-1x224x56x56xf32, 224x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(
            multiply_0, parameter_145, [2, 2], [1, 1], "EXPLICIT", 224, [1, 1], "NCHW"
        )
        del multiply_0, parameter_145

        # pd_op.batch_norm_: (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32, -1xui8) <- (-1x224x28x28xf32, 224xf32, 224xf32, 224xf32, 224xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_0,
                parameter_144,
                parameter_143,
                parameter_142,
                parameter_141,
                True,
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
        del (
            depthwise_conv2d_0,
            parameter_141,
            parameter_142,
            parameter_143,
            parameter_144,
        )

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x224x28x28xf32, 128x224x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            batch_norm__54, parameter_140, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_140

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_139,
                parameter_138,
                parameter_137,
                parameter_136,
                True,
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
        del conv2d_10, parameter_136, parameter_137, parameter_138, parameter_139

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            relu_9, parameter_135, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_135

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_134,
                parameter_133,
                parameter_132,
                parameter_131,
                True,
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
        del conv2d_11, parameter_131, parameter_132, parameter_133, parameter_134

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            relu_10, parameter_130, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_130

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_129,
                parameter_128,
                parameter_127,
                parameter_126,
                True,
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
        del conv2d_12, parameter_126, parameter_127, parameter_128, parameter_129

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            relu_11, parameter_125, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_125

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_124,
                parameter_123,
                parameter_122,
                parameter_121,
                True,
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
        del conv2d_13, parameter_121, parameter_122, parameter_123, parameter_124

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x128x28x28xf32, 128x128x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_12, parameter_120, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_120

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_119,
                parameter_118,
                parameter_117,
                parameter_116,
                True,
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
        del conv2d_14, parameter_116, parameter_117, parameter_118, parameter_119

        # pd_op.relu: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x224x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32]) <- (-1x224x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32)
        combine_1 = [batch_norm__54, relu_9, relu_10, relu_11, relu_12, relu_13]
        del batch_norm__54, relu_10, relu_11, relu_12, relu_13, relu_9

        # pd_op.concat: (-1x864x28x28xf32) <- ([-1x224x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32, -1x128x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)
        del combine_1, full_1

        # pd_op.conv2d: (-1x448x28x28xf32) <- (-1x864x28x28xf32, 448x864x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_1, parameter_115, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_1, parameter_115

        # pd_op.batch_norm_: (-1x448x28x28xf32, 448xf32, 448xf32, 448xf32, 448xf32, -1xui8) <- (-1x448x28x28xf32, 448xf32, 448xf32, 448xf32, 448xf32)
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
                parameter_114,
                parameter_113,
                parameter_112,
                parameter_111,
                True,
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
        del conv2d_15, parameter_111, parameter_112, parameter_113, parameter_114

        # pd_op.relu: (-1x448x28x28xf32) <- (-1x448x28x28xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [1, 1]

        # pd_op.pool2d: (-1x448x1x1xf32) <- (-1x448x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            relu_14,
            full_int_array_3,
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
        del full_int_array_3

        # pd_op.conv2d: (-1x448x1x1xf32) <- (-1x448x1x1xf32, 448x448x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            pool2d_2, parameter_110, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_110, pool2d_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, -1, 1, 1]

        # pd_op.reshape: (1x448x1x1xf32) <- (448xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(parameter_109, full_int_array_4)
        del full_int_array_4, parameter_109

        # pd_op.add: (-1x448x1x1xf32) <- (-1x448x1x1xf32, 1x448x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_16, reshape_1)
        del conv2d_16, reshape_1

        # pd_op.sigmoid: (-1x448x1x1xf32) <- (-1x448x1x1xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(add_2)
        del add_2

        # pd_op.multiply: (-1x448x28x28xf32) <- (-1x448x28x28xf32, -1x448x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(relu_14, sigmoid_1)
        del relu_14, sigmoid_1

        # pd_op.depthwise_conv2d: (-1x448x14x14xf32) <- (-1x448x28x28xf32, 448x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(
            multiply_1, parameter_108, [2, 2], [1, 1], "EXPLICIT", 448, [1, 1], "NCHW"
        )
        del multiply_1, parameter_108

        # pd_op.batch_norm_: (-1x448x14x14xf32, 448xf32, 448xf32, 448xf32, 448xf32, -1xui8) <- (-1x448x14x14xf32, 448xf32, 448xf32, 448xf32, 448xf32)
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
                parameter_107,
                parameter_106,
                parameter_105,
                parameter_104,
                True,
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
        del (
            depthwise_conv2d_1,
            parameter_104,
            parameter_105,
            parameter_106,
            parameter_107,
        )

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x448x14x14xf32, 160x448x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            batch_norm__96, parameter_103, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_103

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_102,
                parameter_101,
                parameter_100,
                parameter_99,
                True,
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
        del conv2d_17, parameter_100, parameter_101, parameter_102, parameter_99

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_15, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_97,
                parameter_96,
                parameter_95,
                parameter_94,
                True,
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
        del conv2d_18, parameter_94, parameter_95, parameter_96, parameter_97

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_16, parameter_93, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_93

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_92,
                parameter_91,
                parameter_90,
                parameter_89,
                True,
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
        del conv2d_19, parameter_89, parameter_90, parameter_91, parameter_92

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__114)
        del batch_norm__114

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_17, parameter_88, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_87,
                parameter_86,
                parameter_85,
                parameter_84,
                True,
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
        del conv2d_20, parameter_84, parameter_85, parameter_86, parameter_87

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_18, parameter_83, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_82,
                parameter_81,
                parameter_80,
                parameter_79,
                True,
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
        del conv2d_21, parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_19 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x448x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32]) <- (-1x448x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32)
        combine_2 = [batch_norm__96, relu_15, relu_16, relu_17, relu_18, relu_19]
        del batch_norm__96, relu_15, relu_16, relu_17, relu_18, relu_19

        # pd_op.concat: (-1x1248x14x14xf32) <- ([-1x448x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)
        del combine_2, full_2

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1248x14x14xf32, 512x1248x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_2, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_2, parameter_78

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
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
                parameter_77,
                parameter_76,
                parameter_75,
                parameter_74,
                True,
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
        del conv2d_22, parameter_74, parameter_75, parameter_76, parameter_77

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__132)
        del batch_norm__132

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(
            relu_20,
            full_int_array_5,
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
        del full_int_array_5

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 512x512x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            pool2d_3, parameter_73, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_73, pool2d_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, -1, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_72, full_int_array_6)
        del full_int_array_6, parameter_72

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_23, reshape_2)
        del conv2d_23, reshape_2

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(add_3)
        del add_3

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(relu_20, sigmoid_2)
        del relu_20, sigmoid_2

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x512x14x14xf32, 160x512x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            multiply_2, parameter_71, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_24,
                parameter_70,
                parameter_69,
                parameter_68,
                parameter_67,
                True,
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
        del conv2d_24, parameter_67, parameter_68, parameter_69, parameter_70

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_21 = paddle._C_ops.relu(batch_norm__138)
        del batch_norm__138

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            relu_21, parameter_66, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_66

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
                parameter_65,
                parameter_64,
                parameter_63,
                parameter_62,
                True,
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
        del conv2d_25, parameter_62, parameter_63, parameter_64, parameter_65

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__144)
        del batch_norm__144

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_22, parameter_61, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_61

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_26,
                parameter_60,
                parameter_59,
                parameter_58,
                parameter_57,
                True,
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
        del conv2d_26, parameter_57, parameter_58, parameter_59, parameter_60

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_23 = paddle._C_ops.relu(batch_norm__150)
        del batch_norm__150

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            relu_23, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_27,
                parameter_55,
                parameter_54,
                parameter_53,
                parameter_52,
                True,
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
        del conv2d_27, parameter_52, parameter_53, parameter_54, parameter_55

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__156)
        del batch_norm__156

        # pd_op.conv2d: (-1x160x14x14xf32) <- (-1x160x14x14xf32, 160x160x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            relu_24, parameter_51, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_51

        # pd_op.batch_norm_: (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (-1x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        (
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
                parameter_50,
                parameter_49,
                parameter_48,
                parameter_47,
                True,
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
        del conv2d_28, parameter_47, parameter_48, parameter_49, parameter_50

        # pd_op.relu: (-1x160x14x14xf32) <- (-1x160x14x14xf32)
        relu_25 = paddle._C_ops.relu(batch_norm__162)
        del batch_norm__162

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x512x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32]) <- (-1x512x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32)
        combine_3 = [multiply_2, relu_21, relu_22, relu_23, relu_24, relu_25]
        del relu_21, relu_22, relu_23, relu_24, relu_25

        # pd_op.concat: (-1x1312x14x14xf32) <- ([-1x512x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32, -1x160x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)
        del combine_3, full_3

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1312x14x14xf32, 512x1312x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            concat_3, parameter_46, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_3, parameter_46

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_29,
                parameter_45,
                parameter_44,
                parameter_43,
                parameter_42,
                True,
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
        del conv2d_29, parameter_42, parameter_43, parameter_44, parameter_45

        # pd_op.relu: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__168)
        del batch_norm__168

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x14x14xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(
            relu_26,
            full_int_array_7,
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
        del full_int_array_7

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 512x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            pool2d_4, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41, pool2d_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, -1, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32) <- (512xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(parameter_40, full_int_array_8)
        del full_int_array_8, parameter_40

        # pd_op.add: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_30, reshape_3)
        del conv2d_30, reshape_3

        # pd_op.sigmoid: (-1x512x1x1xf32) <- (-1x512x1x1xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(add_4)
        del add_4

        # pd_op.multiply: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(relu_26, sigmoid_3)
        del relu_26, sigmoid_3

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_5 = paddle._C_ops.add(multiply_3, multiply_2)
        del multiply_2, multiply_3

        # pd_op.depthwise_conv2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 512x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(
            add_5, parameter_39, [2, 2], [1, 1], "EXPLICIT", 512, [1, 1], "NCHW"
        )
        del add_5, parameter_39

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                depthwise_conv2d_2,
                parameter_38,
                parameter_37,
                parameter_36,
                parameter_35,
                True,
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
        del depthwise_conv2d_2, parameter_35, parameter_36, parameter_37, parameter_38

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x512x7x7xf32, 192x512x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            batch_norm__174, parameter_34, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
                parameter_33,
                parameter_32,
                parameter_31,
                parameter_30,
                True,
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
        del conv2d_31, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (-1x192x7x7xf32) <- (-1x192x7x7xf32)
        relu_27 = paddle._C_ops.relu(batch_norm__180)
        del batch_norm__180

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_27, parameter_29, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_32,
                parameter_28,
                parameter_27,
                parameter_26,
                parameter_25,
                True,
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
        del conv2d_32, parameter_25, parameter_26, parameter_27, parameter_28

        # pd_op.relu: (-1x192x7x7xf32) <- (-1x192x7x7xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__186)
        del batch_norm__186

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            relu_28, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_24

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_33,
                parameter_23,
                parameter_22,
                parameter_21,
                parameter_20,
                True,
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
        del conv2d_33, parameter_20, parameter_21, parameter_22, parameter_23

        # pd_op.relu: (-1x192x7x7xf32) <- (-1x192x7x7xf32)
        relu_29 = paddle._C_ops.relu(batch_norm__192)
        del batch_norm__192

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            relu_29, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_19

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
                parameter_18,
                parameter_17,
                parameter_16,
                parameter_15,
                True,
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
        del conv2d_34, parameter_15, parameter_16, parameter_17, parameter_18

        # pd_op.relu: (-1x192x7x7xf32) <- (-1x192x7x7xf32)
        relu_30 = paddle._C_ops.relu(batch_norm__198)
        del batch_norm__198

        # pd_op.conv2d: (-1x192x7x7xf32) <- (-1x192x7x7xf32, 192x192x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_30, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_14

        # pd_op.batch_norm_: (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (-1x192x7x7xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        (
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_35,
                parameter_13,
                parameter_12,
                parameter_11,
                parameter_10,
                True,
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
        del conv2d_35, parameter_10, parameter_11, parameter_12, parameter_13

        # pd_op.relu: (-1x192x7x7xf32) <- (-1x192x7x7xf32)
        relu_31 = paddle._C_ops.relu(batch_norm__204)
        del batch_norm__204

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x512x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32]) <- (-1x512x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32)
        combine_4 = [batch_norm__174, relu_27, relu_28, relu_29, relu_30, relu_31]
        del batch_norm__174, relu_27, relu_28, relu_29, relu_30, relu_31

        # pd_op.concat: (-1x1472x7x7xf32) <- ([-1x512x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32, -1x192x7x7xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_4)
        del combine_4, full_4

        # pd_op.conv2d: (-1x768x7x7xf32) <- (-1x1472x7x7xf32, 768x1472x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            concat_4, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_4, parameter_9

        # pd_op.batch_norm_: (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32, -1xui8) <- (-1x768x7x7xf32, 768xf32, 768xf32, 768xf32, 768xf32)
        (
            batch_norm__210,
            batch_norm__211,
            batch_norm__212,
            batch_norm__213,
            batch_norm__214,
            batch_norm__215,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_36,
                parameter_8,
                parameter_7,
                parameter_6,
                parameter_5,
                True,
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
        del conv2d_36, parameter_5, parameter_6, parameter_7, parameter_8

        # pd_op.relu: (-1x768x7x7xf32) <- (-1x768x7x7xf32)
        relu_32 = paddle._C_ops.relu(batch_norm__210)
        del batch_norm__210

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [1, 1]

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x7x7xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(
            relu_32,
            full_int_array_9,
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
        del full_int_array_9

        # pd_op.conv2d: (-1x768x1x1xf32) <- (-1x768x1x1xf32, 768x768x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            pool2d_5, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_4, pool2d_5

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, -1, 1, 1]

        # pd_op.reshape: (1x768x1x1xf32) <- (768xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_3, full_int_array_10)
        del full_int_array_10, parameter_3

        # pd_op.add: (-1x768x1x1xf32) <- (-1x768x1x1xf32, 1x768x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_37, reshape_4)
        del conv2d_37, reshape_4

        # pd_op.sigmoid: (-1x768x1x1xf32) <- (-1x768x1x1xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(add_6)
        del add_6

        # pd_op.multiply: (-1x768x7x7xf32) <- (-1x768x7x7xf32, -1x768x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(relu_32, sigmoid_4)
        del relu_32, sigmoid_4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [1, 1]

        # pd_op.pool2d: (-1x768x1x1xf32) <- (-1x768x7x7xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(
            multiply_4,
            full_int_array_11,
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
        del full_int_array_11, multiply_4

        # pd_op.conv2d: (-1x2048x1x1xf32) <- (-1x768x1x1xf32, 2048x768x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            pool2d_6, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_6

        # pd_op.relu: (-1x2048x1x1xf32) <- (-1x2048x1x1xf32)
        relu_33 = paddle._C_ops.relu(conv2d_38)
        del conv2d_38

        # pd_op.flatten: (-1x2048xf32) <- (-1x2048x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_33, 1, 3)
        del relu_33

        # pd_op.matmul: (-1x102xf32) <- (-1x2048xf32, 2048x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
