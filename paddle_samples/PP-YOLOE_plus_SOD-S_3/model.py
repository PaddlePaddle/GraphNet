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
        data_0,
        data_1,
        data_2,
    ):
        # pd_op.transpose: (2x512x324xf32) <- (2x324x512xf32)
        transpose_0 = paddle._C_ops.transpose(data_0, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [2, 512, 18, 18]

        # pd_op.reshape: (2x512x18x18xf32) <- (2x512x324xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, full_int_array_3)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x512x18x18xf32, 192x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            reshape_0, parameter_174, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x512x18x18xf32, 192x512x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            reshape_0, parameter_169, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            swish_1, parameter_164, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            swish_2, parameter_159, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            swish_2, parameter_154, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.add: (2x192x18x18xf32) <- (2x192x18x18xf32, 2x192x18x18xf32)
        add_0 = paddle._C_ops.add(batch_norm__18, batch_norm__24)

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_3 = paddle._C_ops.swish(add_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [5, 5]

        # pd_op.pool2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            swish_3,
            full_int_array_0,
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
        full_int_array_1 = [9, 9]

        # pd_op.pool2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(
            swish_3,
            full_int_array_1,
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
        full_int_array_2 = [13, 13]

        # pd_op.pool2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(
            swish_3,
            full_int_array_2,
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

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

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

        # builtin.combine: ([2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32]) <- (2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32)
        combine_0 = [swish_3, pool2d_0, pool2d_1, pool2d_2]

        # pd_op.concat: (2x768x18x18xf32) <- ([2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32, 2x192x18x18xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x768x18x18xf32, 192x768x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            concat_0, parameter_149, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__30)

        # builtin.combine: ([2x192x18x18xf32, 2x192x18x18xf32]) <- (2x192x18x18xf32, 2x192x18x18xf32)
        combine_1 = [swish_0, swish_4]

        # pd_op.concat: (2x384x18x18xf32) <- ([2x192x18x18xf32, 2x192x18x18xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.conv2d: (2x384x18x18xf32) <- (2x384x18x18xf32, 384x384x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_144, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x384x18x18xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x18x18xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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

        # pd_op.swish: (2x384x18x18xf32) <- (2x384x18x18xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x384x18x18xf32, 192x384x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            swish_5, parameter_139, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__42)

        # pd_op.nearest_interp: (2x192x36x36xf32) <- (2x192x18x18xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(
            swish_6,
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

        # builtin.combine: ([2x192x36x36xf32, 2x256x36x36xf32]) <- (2x192x36x36xf32, 2x256x36x36xf32)
        combine_2 = [nearest_interp_0, data_2]

        # pd_op.concat: (2x448x36x36xf32) <- ([2x192x36x36xf32, 2x256x36x36xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x448x36x36xf32, 96x448x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            concat_2, parameter_134, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_7 = paddle._C_ops.swish(batch_norm__48)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x448x36x36xf32, 96x448x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_129, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_8 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            swish_8, parameter_124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            swish_9, parameter_119, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            swish_9, parameter_114, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.add: (2x96x36x36xf32) <- (2x96x36x36xf32, 2x96x36x36xf32)
        add_1 = paddle._C_ops.add(batch_norm__66, batch_norm__72)

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_10 = paddle._C_ops.swish(add_1)

        # builtin.combine: ([2x96x36x36xf32, 2x96x36x36xf32]) <- (2x96x36x36xf32, 2x96x36x36xf32)
        combine_3 = [swish_7, swish_10]

        # pd_op.concat: (2x192x36x36xf32) <- ([2x96x36x36xf32, 2x96x36x36xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.conv2d: (2x192x36x36xf32) <- (2x192x36x36xf32, 192x192x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_109, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x36x36xf32) <- (2x192x36x36xf32)
        swish_11 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x192x36x36xf32, 96x192x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            swish_11, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__84)

        # pd_op.nearest_interp: (2x96x72x72xf32) <- (2x96x36x36xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(
            swish_12,
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

        # builtin.combine: ([2x96x72x72xf32, 2x128x72x72xf32]) <- (2x96x72x72xf32, 2x128x72x72xf32)
        combine_4 = [nearest_interp_1, data_1]

        # pd_op.concat: (2x224x72x72xf32) <- ([2x96x72x72xf32, 2x128x72x72xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)

        # pd_op.conv2d: (2x48x72x72xf32) <- (2x224x72x72xf32, 48x224x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            concat_4, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.swish: (2x48x72x72xf32) <- (2x48x72x72xf32)
        swish_13 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.conv2d: (2x48x72x72xf32) <- (2x224x72x72xf32, 48x224x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_94, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.swish: (2x48x72x72xf32) <- (2x48x72x72xf32)
        swish_14 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.conv2d: (2x48x72x72xf32) <- (2x48x72x72xf32, 48x48x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            swish_14, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.swish: (2x48x72x72xf32) <- (2x48x72x72xf32)
        swish_15 = paddle._C_ops.swish(batch_norm__102)

        # pd_op.conv2d: (2x48x72x72xf32) <- (2x48x72x72xf32, 48x48x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            swish_15, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.conv2d: (2x48x72x72xf32) <- (2x48x72x72xf32, 48x48x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            swish_15, parameter_79, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32, -1xui8) <- (2x48x72x72xf32, 48xf32, 48xf32, 48xf32, 48xf32)
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

        # pd_op.add: (2x48x72x72xf32) <- (2x48x72x72xf32, 2x48x72x72xf32)
        add_2 = paddle._C_ops.add(batch_norm__108, batch_norm__114)

        # pd_op.swish: (2x48x72x72xf32) <- (2x48x72x72xf32)
        swish_16 = paddle._C_ops.swish(add_2)

        # builtin.combine: ([2x48x72x72xf32, 2x48x72x72xf32]) <- (2x48x72x72xf32, 2x48x72x72xf32)
        combine_5 = [swish_13, swish_16]

        # pd_op.concat: (2x96x72x72xf32) <- ([2x48x72x72xf32, 2x48x72x72xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)

        # pd_op.conv2d: (2x96x72x72xf32) <- (2x96x72x72xf32, 96x96x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            concat_5, parameter_74, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x72x72xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x72x72xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x72x72xf32) <- (2x96x72x72xf32)
        swish_29 = paddle._C_ops.swish(batch_norm__120)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x72x72xf32, 96x96x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            swish_29, parameter_69, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_17 = paddle._C_ops.swish(batch_norm__126)

        # builtin.combine: ([2x96x36x36xf32, 2x192x36x36xf32]) <- (2x96x36x36xf32, 2x192x36x36xf32)
        combine_6 = [swish_17, swish_11]

        # pd_op.concat: (2x288x36x36xf32) <- ([2x96x36x36xf32, 2x192x36x36xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x288x36x36xf32, 96x288x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x288x36x36xf32, 96x288x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            concat_6, parameter_59, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__138)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            swish_19, parameter_54, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_20 = paddle._C_ops.swish(batch_norm__144)

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            swish_20, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.conv2d: (2x96x36x36xf32) <- (2x96x36x36xf32, 96x96x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            swish_20, parameter_44, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32, -1xui8) <- (2x96x36x36xf32, 96xf32, 96xf32, 96xf32, 96xf32)
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

        # pd_op.add: (2x96x36x36xf32) <- (2x96x36x36xf32, 2x96x36x36xf32)
        add_3 = paddle._C_ops.add(batch_norm__150, batch_norm__156)

        # pd_op.swish: (2x96x36x36xf32) <- (2x96x36x36xf32)
        swish_21 = paddle._C_ops.swish(add_3)

        # builtin.combine: ([2x96x36x36xf32, 2x96x36x36xf32]) <- (2x96x36x36xf32, 2x96x36x36xf32)
        combine_7 = [swish_18, swish_21]

        # pd_op.concat: (2x192x36x36xf32) <- ([2x96x36x36xf32, 2x96x36x36xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)

        # pd_op.conv2d: (2x192x36x36xf32) <- (2x192x36x36xf32, 192x192x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            concat_7, parameter_39, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x36x36xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x36x36xf32) <- (2x192x36x36xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x36x36xf32, 192x192x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            swish_28, parameter_34, [2, 2], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__168)

        # builtin.combine: ([2x192x18x18xf32, 2x384x18x18xf32]) <- (2x192x18x18xf32, 2x384x18x18xf32)
        combine_8 = [swish_22, swish_5]

        # pd_op.concat: (2x576x18x18xf32) <- ([2x192x18x18xf32, 2x384x18x18xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x576x18x18xf32, 192x576x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            concat_8, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_23 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x576x18x18xf32, 192x576x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            concat_8, parameter_24, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_24 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            swish_24, parameter_19, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            swish_25, parameter_14, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.conv2d: (2x192x18x18xf32) <- (2x192x18x18xf32, 192x192x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            swish_25, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32, -1xui8) <- (2x192x18x18xf32, 192xf32, 192xf32, 192xf32, 192xf32)
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

        # pd_op.add: (2x192x18x18xf32) <- (2x192x18x18xf32, 2x192x18x18xf32)
        add_4 = paddle._C_ops.add(batch_norm__192, batch_norm__198)

        # pd_op.swish: (2x192x18x18xf32) <- (2x192x18x18xf32)
        swish_26 = paddle._C_ops.swish(add_4)

        # builtin.combine: ([2x192x18x18xf32, 2x192x18x18xf32]) <- (2x192x18x18xf32, 2x192x18x18xf32)
        combine_9 = [swish_23, swish_26]

        # pd_op.concat: (2x384x18x18xf32) <- ([2x192x18x18xf32, 2x192x18x18xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)

        # pd_op.conv2d: (2x384x18x18xf32) <- (2x384x18x18xf32, 384x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_9, parameter_4, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (2x384x18x18xf32, 384xf32, 384xf32, 384xf32, 384xf32, -1xui8) <- (2x384x18x18xf32, 384xf32, 384xf32, 384xf32, 384xf32)
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

        # pd_op.swish: (2x384x18x18xf32) <- (2x384x18x18xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__204)
        return (
            transpose_0,
            conv2d_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            batch_norm__5,
            swish_0,
            conv2d_1,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
            batch_norm__11,
            swish_1,
            conv2d_2,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            swish_2,
            conv2d_3,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            conv2d_4,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            add_0,
            swish_3,
            full_int_array_0,
            pool2d_0,
            full_int_array_1,
            pool2d_1,
            full_int_array_2,
            pool2d_2,
            full_0,
            concat_0,
            conv2d_5,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            swish_4,
            assign_0,
            concat_1,
            conv2d_6,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            swish_5,
            conv2d_7,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            swish_6,
            nearest_interp_0,
            assign_1,
            concat_2,
            conv2d_8,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            swish_7,
            conv2d_9,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            swish_8,
            conv2d_10,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            swish_9,
            conv2d_11,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            conv2d_12,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            add_1,
            swish_10,
            assign_2,
            concat_3,
            conv2d_13,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
            batch_norm__83,
            swish_11,
            conv2d_14,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
            swish_12,
            nearest_interp_1,
            assign_3,
            concat_4,
            conv2d_15,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            swish_13,
            conv2d_16,
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            swish_14,
            conv2d_17,
            batch_norm__102,
            batch_norm__103,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            swish_15,
            conv2d_18,
            batch_norm__108,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            conv2d_19,
            batch_norm__114,
            batch_norm__115,
            batch_norm__116,
            batch_norm__117,
            batch_norm__118,
            batch_norm__119,
            add_2,
            swish_16,
            assign_4,
            concat_5,
            conv2d_20,
            batch_norm__120,
            batch_norm__121,
            batch_norm__122,
            batch_norm__123,
            batch_norm__124,
            batch_norm__125,
            conv2d_21,
            batch_norm__126,
            batch_norm__127,
            batch_norm__128,
            batch_norm__129,
            batch_norm__130,
            batch_norm__131,
            swish_17,
            assign_5,
            concat_6,
            conv2d_22,
            batch_norm__132,
            batch_norm__133,
            batch_norm__134,
            batch_norm__135,
            batch_norm__136,
            batch_norm__137,
            swish_18,
            conv2d_23,
            batch_norm__138,
            batch_norm__139,
            batch_norm__140,
            batch_norm__141,
            batch_norm__142,
            batch_norm__143,
            swish_19,
            conv2d_24,
            batch_norm__144,
            batch_norm__145,
            batch_norm__146,
            batch_norm__147,
            batch_norm__148,
            batch_norm__149,
            swish_20,
            conv2d_25,
            batch_norm__150,
            batch_norm__151,
            batch_norm__152,
            batch_norm__153,
            batch_norm__154,
            batch_norm__155,
            conv2d_26,
            batch_norm__156,
            batch_norm__157,
            batch_norm__158,
            batch_norm__159,
            batch_norm__160,
            batch_norm__161,
            add_3,
            swish_21,
            assign_6,
            concat_7,
            conv2d_27,
            batch_norm__162,
            batch_norm__163,
            batch_norm__164,
            batch_norm__165,
            batch_norm__166,
            batch_norm__167,
            conv2d_28,
            batch_norm__168,
            batch_norm__169,
            batch_norm__170,
            batch_norm__171,
            batch_norm__172,
            batch_norm__173,
            swish_22,
            assign_7,
            concat_8,
            conv2d_29,
            batch_norm__174,
            batch_norm__175,
            batch_norm__176,
            batch_norm__177,
            batch_norm__178,
            batch_norm__179,
            swish_23,
            conv2d_30,
            batch_norm__180,
            batch_norm__181,
            batch_norm__182,
            batch_norm__183,
            batch_norm__184,
            batch_norm__185,
            swish_24,
            conv2d_31,
            batch_norm__186,
            batch_norm__187,
            batch_norm__188,
            batch_norm__189,
            batch_norm__190,
            batch_norm__191,
            swish_25,
            conv2d_32,
            batch_norm__192,
            batch_norm__193,
            batch_norm__194,
            batch_norm__195,
            batch_norm__196,
            batch_norm__197,
            conv2d_33,
            batch_norm__198,
            batch_norm__199,
            batch_norm__200,
            batch_norm__201,
            batch_norm__202,
            batch_norm__203,
            add_4,
            swish_26,
            assign_8,
            concat_9,
            conv2d_34,
            batch_norm__204,
            batch_norm__205,
            batch_norm__206,
            batch_norm__207,
            batch_norm__208,
            batch_norm__209,
            swish_27,
            swish_28,
            swish_29,
            reshape_0,
        )
