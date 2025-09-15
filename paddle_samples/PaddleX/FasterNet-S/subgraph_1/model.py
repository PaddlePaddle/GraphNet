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
        data_0,
    ):
        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x3x224x224xf32, 128x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_148, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_148

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
                parameter_147,
                parameter_146,
                parameter_145,
                parameter_144,
                True,
                float("0.1"),
                float("1e-05"),
                "NCHW",
                True,
                False,
            ),
            lambda out: out
            if isinstance(out, (list, tuple))
            else (out, None, None, None, None, None),
        )
        del conv2d_0, parameter_144, parameter_145, parameter_146, parameter_147

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [32, 96]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x32x56x56xf32, -1x96x56x56xf32]) <- (-1x128x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (-1x32x56x56xf32, -1x96x56x56xf32) <- ([-1x32x56x56xf32, -1x96x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (-1x32x56x56xf32) <- (-1x32x56x56xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_143, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_143, split_1

        # builtin.combine: ([-1x32x56x56xf32, -1x96x56x56xf32]) <- (-1x32x56x56xf32, -1x96x56x56xf32)
        combine_0 = [conv2d_1, split_2]
        del conv2d_1, split_2

        # pd_op.concat: (-1x128x56x56xf32) <- ([-1x32x56x56xf32, -1x96x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x128x56x56xf32, 256x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_0, parameter_142

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
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
                parameter_141,
                parameter_140,
                parameter_139,
                parameter_138,
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
        del conv2d_2, parameter_138, parameter_139, parameter_140, parameter_141

        # pd_op.relu: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__6)
        del batch_norm__6

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_137, relu_0

        # pd_op.add: (-1x128x56x56xf32) <- (-1x128x56x56xf32, -1x128x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)
        del batch_norm__0, conv2d_3

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x128x56x56xf32, 256x128x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_1, parameter_136, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_1, parameter_136

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
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
        del conv2d_4, parameter_132, parameter_133, parameter_134, parameter_135

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [64, 192]

        # pd_op.split: ([-1x64x28x28xf32, -1x192x28x28xf32]) <- (-1x256x28x28xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(batch_norm__12, full_int_array_1, full_0)

        # builtin.split: (-1x64x28x28xf32, -1x192x28x28xf32) <- ([-1x64x28x28xf32, -1x192x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_4, parameter_131, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_131, split_4

        # builtin.combine: ([-1x64x28x28xf32, -1x192x28x28xf32]) <- (-1x64x28x28xf32, -1x192x28x28xf32)
        combine_1 = [conv2d_5, split_5]
        del conv2d_5, split_5

        # pd_op.concat: (-1x256x28x28xf32) <- ([-1x64x28x28xf32, -1x192x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_1, parameter_130

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
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
        del conv2d_6, parameter_126, parameter_127, parameter_128, parameter_129

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__18)
        del batch_norm__18

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_125, relu_1

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        add_2 = paddle._C_ops.add(batch_norm__12, conv2d_7)
        del batch_norm__12, conv2d_7

        # pd_op.split: ([-1x64x28x28xf32, -1x192x28x28xf32]) <- (-1x256x28x28xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_2, full_int_array_1, full_0)
        del full_int_array_1

        # builtin.split: (-1x64x28x28xf32, -1x192x28x28xf32) <- ([-1x64x28x28xf32, -1x192x28x28xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (-1x64x28x28xf32) <- (-1x64x28x28xf32, 64x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_7, parameter_124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_124, split_7

        # builtin.combine: ([-1x64x28x28xf32, -1x192x28x28xf32]) <- (-1x64x28x28xf32, -1x192x28x28xf32)
        combine_2 = [conv2d_8, split_8]
        del conv2d_8, split_8

        # pd_op.concat: (-1x256x28x28xf32) <- ([-1x64x28x28xf32, -1x192x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x28x28xf32, 512x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_2, parameter_123

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_122,
                parameter_121,
                parameter_120,
                parameter_119,
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
        del conv2d_9, parameter_119, parameter_120, parameter_121, parameter_122

        # pd_op.relu: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__24)
        del batch_norm__24

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_118, relu_2

        # pd_op.add: (-1x256x28x28xf32) <- (-1x256x28x28xf32, -1x256x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, conv2d_10)
        del add_2, conv2d_10

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x256x28x28xf32, 512x256x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_3, parameter_117, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_3, parameter_117

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
                parameter_116,
                parameter_115,
                parameter_114,
                parameter_113,
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
        del conv2d_11, parameter_113, parameter_114, parameter_115, parameter_116

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [128, 384]

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__30, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_10, parameter_112, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_112, split_10

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_3 = [conv2d_12, split_11]
        del conv2d_12, split_11

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_3, parameter_111

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
                parameter_110,
                parameter_109,
                parameter_108,
                parameter_107,
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
        del conv2d_13, parameter_107, parameter_108, parameter_109, parameter_110

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__36)
        del batch_norm__36

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_3, parameter_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_106, relu_3

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__30, conv2d_14)
        del batch_norm__30, conv2d_14

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_4, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_13, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_105, split_13

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_4 = [conv2d_15, split_14]
        del conv2d_15, split_14

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_4, parameter_104

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
                parameter_103,
                parameter_102,
                parameter_101,
                parameter_100,
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
        del conv2d_16, parameter_100, parameter_101, parameter_102, parameter_103

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__42)
        del batch_norm__42

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_4, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_99, relu_4

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, conv2d_17)
        del add_4, conv2d_17

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_5, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_16, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_98, split_16

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_5 = [conv2d_18, split_17]
        del conv2d_18, split_17

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_5, parameter_97

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
                parameter_96,
                parameter_95,
                parameter_94,
                parameter_93,
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
        del conv2d_19, parameter_93, parameter_94, parameter_95, parameter_96

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__48)
        del batch_norm__48

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_5, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_92, relu_5

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, conv2d_20)
        del add_5, conv2d_20

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_6, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_19, parameter_91, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_91, split_19

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_6 = [conv2d_21, split_20]
        del conv2d_21, split_20

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_6, parameter_90

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
                parameter_89,
                parameter_88,
                parameter_87,
                parameter_86,
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
        del conv2d_22, parameter_86, parameter_87, parameter_88, parameter_89

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__54)
        del batch_norm__54

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_6, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_85, relu_6

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, conv2d_23)
        del add_6, conv2d_23

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(add_7, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_84, split_22

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_7 = [conv2d_24, split_23]
        del conv2d_24, split_23

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_7, parameter_83

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
        del conv2d_25, parameter_79, parameter_80, parameter_81, parameter_82

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__60)
        del batch_norm__60

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_78, relu_7

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, conv2d_26)
        del add_7, conv2d_26

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_8, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77, split_25

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_8 = [conv2d_27, split_26]
        del conv2d_27, split_26

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)
        del combine_8

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_8, parameter_76

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_75,
                parameter_74,
                parameter_73,
                parameter_72,
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
        del conv2d_28, parameter_72, parameter_73, parameter_74, parameter_75

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__66)
        del batch_norm__66

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71, relu_8

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, conv2d_29)
        del add_8, conv2d_29

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_9, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70, split_28

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_9 = [conv2d_30, split_29]
        del conv2d_30, split_29

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_9, parameter_69

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_68,
                parameter_67,
                parameter_66,
                parameter_65,
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
        del conv2d_31, parameter_65, parameter_66, parameter_67, parameter_68

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__72)
        del batch_norm__72

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64, relu_9

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, conv2d_32)
        del add_9, conv2d_32

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_10, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63, split_31

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_10 = [conv2d_33, split_32]
        del conv2d_33, split_32

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)
        del combine_10

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_10, parameter_62

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_61,
                parameter_60,
                parameter_59,
                parameter_58,
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
        del conv2d_34, parameter_58, parameter_59, parameter_60, parameter_61

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__78)
        del batch_norm__78

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57, relu_10

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, conv2d_35)
        del add_10, conv2d_35

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(add_11, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            split_34, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56, split_34

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_11 = [conv2d_36, split_35]
        del conv2d_36, split_35

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            concat_11, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_11, parameter_55

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_54,
                parameter_53,
                parameter_52,
                parameter_51,
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
        del conv2d_37, parameter_51, parameter_52, parameter_53, parameter_54

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__84)
        del batch_norm__84

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_11, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50, relu_11

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_12 = paddle._C_ops.add(add_11, conv2d_38)
        del add_11, conv2d_38

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_12, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            split_37, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49, split_37

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_12 = [conv2d_39, split_38]
        del conv2d_39, split_38

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)
        del combine_12

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_12, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_12, parameter_48

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_47,
                parameter_46,
                parameter_45,
                parameter_44,
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
        del conv2d_40, parameter_44, parameter_45, parameter_46, parameter_47

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__90)
        del batch_norm__90

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_12, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43, relu_12

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_13 = paddle._C_ops.add(add_12, conv2d_41)
        del add_12, conv2d_41

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_39 = paddle._C_ops.split(add_13, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_40,
            split_41,
        ) = split_39
        del split_39

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            split_40, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42, split_40

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_13 = [conv2d_42, split_41]
        del conv2d_42, split_41

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_0)
        del combine_13

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            concat_13, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_13, parameter_41

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_40,
                parameter_39,
                parameter_38,
                parameter_37,
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
        del conv2d_43, parameter_37, parameter_38, parameter_39, parameter_40

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__96)
        del batch_norm__96

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_13, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36, relu_13

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_14 = paddle._C_ops.add(add_13, conv2d_44)
        del add_13, conv2d_44

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(add_14, full_int_array_2, full_0)

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_43,
            split_44,
        ) = split_42
        del split_42

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            split_43, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35, split_43

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_14 = [conv2d_45, split_44]
        del conv2d_45, split_44

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_0)
        del combine_14

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            concat_14, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_14, parameter_34

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
        del conv2d_46, parameter_30, parameter_31, parameter_32, parameter_33

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__102)
        del batch_norm__102

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_14, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29, relu_14

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_15 = paddle._C_ops.add(add_14, conv2d_47)
        del add_14, conv2d_47

        # pd_op.split: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x512x14x14xf32, 2xi64, 1xi32)
        split_45 = paddle._C_ops.split(add_15, full_int_array_2, full_0)
        del full_int_array_2

        # builtin.split: (-1x128x14x14xf32, -1x384x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32])
        (
            split_46,
            split_47,
        ) = split_45
        del split_45

        # pd_op.conv2d: (-1x128x14x14xf32) <- (-1x128x14x14xf32, 128x128x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            split_46, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28, split_46

        # builtin.combine: ([-1x128x14x14xf32, -1x384x14x14xf32]) <- (-1x128x14x14xf32, -1x384x14x14xf32)
        combine_15 = [conv2d_48, split_47]
        del conv2d_48, split_47

        # pd_op.concat: (-1x512x14x14xf32) <- ([-1x128x14x14xf32, -1x384x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_0)
        del combine_15

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            concat_15, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_15, parameter_27

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
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
                parameter_26,
                parameter_25,
                parameter_24,
                parameter_23,
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
        del conv2d_49, parameter_23, parameter_24, parameter_25, parameter_26

        # pd_op.relu: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__108)
        del batch_norm__108

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_15, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22, relu_15

        # pd_op.add: (-1x512x14x14xf32) <- (-1x512x14x14xf32, -1x512x14x14xf32)
        add_16 = paddle._C_ops.add(add_15, conv2d_50)
        del add_15, conv2d_50

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x512x14x14xf32, 1024x512x2x2xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            add_16, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del add_16, parameter_21

        # pd_op.batch_norm_: (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (-1x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_51,
                parameter_20,
                parameter_19,
                parameter_18,
                parameter_17,
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
        del conv2d_51, parameter_17, parameter_18, parameter_19, parameter_20

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [256, 768]

        # pd_op.split: ([-1x256x7x7xf32, -1x768x7x7xf32]) <- (-1x1024x7x7xf32, 2xi64, 1xi32)
        split_48 = paddle._C_ops.split(batch_norm__114, full_int_array_3, full_0)

        # builtin.split: (-1x256x7x7xf32, -1x768x7x7xf32) <- ([-1x256x7x7xf32, -1x768x7x7xf32])
        (
            split_49,
            split_50,
        ) = split_48
        del split_48

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            split_49, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16, split_49

        # builtin.combine: ([-1x256x7x7xf32, -1x768x7x7xf32]) <- (-1x256x7x7xf32, -1x768x7x7xf32)
        combine_16 = [conv2d_52, split_50]
        del conv2d_52, split_50

        # pd_op.concat: (-1x1024x7x7xf32) <- ([-1x256x7x7xf32, -1x768x7x7xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_0)
        del combine_16

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            concat_16, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_16, parameter_15

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        (
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_53,
                parameter_14,
                parameter_13,
                parameter_12,
                parameter_11,
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
        del conv2d_53, parameter_11, parameter_12, parameter_13, parameter_14

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__120)
        del batch_norm__120

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            relu_16, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10, relu_16

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        add_17 = paddle._C_ops.add(batch_norm__114, conv2d_54)
        del batch_norm__114, conv2d_54

        # pd_op.split: ([-1x256x7x7xf32, -1x768x7x7xf32]) <- (-1x1024x7x7xf32, 2xi64, 1xi32)
        split_51 = paddle._C_ops.split(add_17, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (-1x256x7x7xf32, -1x768x7x7xf32) <- ([-1x256x7x7xf32, -1x768x7x7xf32])
        (
            split_52,
            split_53,
        ) = split_51
        del split_51

        # pd_op.conv2d: (-1x256x7x7xf32) <- (-1x256x7x7xf32, 256x256x3x3xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            split_52, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9, split_52

        # builtin.combine: ([-1x256x7x7xf32, -1x768x7x7xf32]) <- (-1x256x7x7xf32, -1x768x7x7xf32)
        combine_17 = [conv2d_55, split_53]
        del conv2d_55, split_53

        # pd_op.concat: (-1x1024x7x7xf32) <- ([-1x256x7x7xf32, -1x768x7x7xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_0)
        del combine_17, full_0

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            concat_17, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del concat_17, parameter_8

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        (
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_56,
                parameter_7,
                parameter_6,
                parameter_5,
                parameter_4,
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
        del conv2d_56, parameter_4, parameter_5, parameter_6, parameter_7

        # pd_op.relu: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__126)
        del batch_norm__126

        # pd_op.conv2d: (-1x1024x7x7xf32) <- (-1x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            relu_17, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3, relu_17

        # pd_op.add: (-1x1024x7x7xf32) <- (-1x1024x7x7xf32, -1x1024x7x7xf32)
        add_18 = paddle._C_ops.add(add_17, conv2d_57)
        del add_17, conv2d_57

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf32) <- (-1x1024x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_18,
            full_int_array_4,
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
        del add_18, full_int_array_4

        # pd_op.conv2d: (-1x1280x1x1xf32) <- (-1x1024x1x1xf32, 1280x1024x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2, pool2d_0

        # pd_op.relu: (-1x1280x1x1xf32) <- (-1x1280x1x1xf32)
        relu_18 = paddle._C_ops.relu(conv2d_58)
        del conv2d_58

        # pd_op.flatten: (-1x1280xf32) <- (-1x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_18, 1, 3)
        del relu_18

        # pd_op.matmul: (-1x102xf32) <- (-1x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del flatten_0, parameter_1

        # pd_op.add: (-1x102xf32) <- (-1x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del matmul_0, parameter_0

        return add_0
