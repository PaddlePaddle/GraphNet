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
        # pd_op.conv2d: (128x128x56x56xf32) <- (128x3x224x224xf32, 128x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_148, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (128x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
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
        full_int_array_1 = [32, 96]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

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

        # pd_op.split: ([128x32x56x56xf32, 128x96x56x56xf32]) <- (128x128x56x56xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(batch_norm__0, full_int_array_1, full_0)

        # builtin.split: (128x32x56x56xf32, 128x96x56x56xf32) <- ([128x32x56x56xf32, 128x96x56x56xf32])
        (
            split_0,
            split_1,
        ) = split_36

        # pd_op.conv2d: (128x32x56x56xf32) <- (128x32x56x56xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_0, parameter_143, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x32x56x56xf32, 128x96x56x56xf32]) <- (128x32x56x56xf32, 128x96x56x56xf32)
        combine_0 = [conv2d_1, split_1]

        # pd_op.concat: (128x128x56x56xf32) <- ([128x32x56x56xf32, 128x96x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (128x256x56x56xf32) <- (128x128x56x56xf32, 256x128x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_142, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__114,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            batch_norm__10,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_2,
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

        # pd_op.relu: (128x256x56x56xf32) <- (128x256x56x56xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (128x128x56x56xf32) <- (128x256x56x56xf32, 128x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            relu_0, parameter_137, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.add: (128x128x56x56xf32) <- (128x128x56x56xf32, 128x128x56x56xf32)
        add_0 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x128x56x56xf32, 256x128x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_0, parameter_136, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (128x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_4,
                parameter_135,
                parameter_134,
                parameter_133,
                parameter_132,
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
        full_int_array_2 = [64, 192]

        # pd_op.split: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x256x28x28xf32, 2xi64, 1xi32)
        split_37 = paddle._C_ops.split(batch_norm__11, full_int_array_2, full_0)

        # builtin.split: (128x64x28x28xf32, 128x192x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32])
        (
            split_2,
            split_3,
        ) = split_37

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x64x28x28xf32, 64x64x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_2, parameter_131, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x64x28x28xf32, 128x192x28x28xf32)
        combine_1 = [conv2d_5, split_3]

        # pd_op.concat: (128x256x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # pd_op.conv2d: (128x512x28x28xf32) <- (128x256x28x28xf32, 512x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_130, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__115,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_6,
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

        # pd_op.relu: (128x512x28x28xf32) <- (128x512x28x28xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__115)

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x512x28x28xf32, 256x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            relu_1, parameter_125, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full(
            [],
            float("0.994118"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [128, 1, 1, 1]

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_19 = paddle._C_ops.add(full_1, uniform_0)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_0 = paddle._C_ops.floor(add_19)

        # pd_op.divide: (128x256x28x28xf32) <- (128x256x28x28xf32, xf32)
        divide_0 = paddle._C_ops.divide(conv2d_7, full_1)

        # pd_op.multiply: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x1x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(divide_0, floor_0)

        # pd_op.add: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x256x28x28xf32)
        add_1 = paddle._C_ops.add(batch_norm__11, multiply_0)

        # pd_op.split: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x256x28x28xf32, 2xi64, 1xi32)
        split_38 = paddle._C_ops.split(add_1, full_int_array_2, full_0)

        # builtin.split: (128x64x28x28xf32, 128x192x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_38

        # pd_op.conv2d: (128x64x28x28xf32) <- (128x64x28x28xf32, 64x64x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_4, parameter_124, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x64x28x28xf32, 128x192x28x28xf32]) <- (128x64x28x28xf32, 128x192x28x28xf32)
        combine_2 = [conv2d_8, split_5]

        # pd_op.concat: (128x256x28x28xf32) <- ([128x64x28x28xf32, 128x192x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)

        # pd_op.conv2d: (128x512x28x28xf32) <- (128x256x28x28xf32, 512x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_123, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__116,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_9,
                parameter_122,
                parameter_121,
                parameter_120,
                parameter_119,
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

        # pd_op.relu: (128x512x28x28xf32) <- (128x512x28x28xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__116)

        # pd_op.conv2d: (128x256x28x28xf32) <- (128x512x28x28xf32, 256x512x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_2, parameter_118, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_2 = paddle._C_ops.full(
            [],
            float("0.988235"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_20 = paddle._C_ops.add(full_2, uniform_1)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_1 = paddle._C_ops.floor(add_20)

        # pd_op.divide: (128x256x28x28xf32) <- (128x256x28x28xf32, xf32)
        divide_1 = paddle._C_ops.divide(conv2d_10, full_2)

        # pd_op.multiply: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x1x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(divide_1, floor_1)

        # pd_op.add: (128x256x28x28xf32) <- (128x256x28x28xf32, 128x256x28x28xf32)
        add_2 = paddle._C_ops.add(add_1, multiply_1)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x256x28x28xf32, 512x256x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_2, parameter_117, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32, -1xui8) <- (128x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        (
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_11,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [128, 384]

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_39 = paddle._C_ops.split(batch_norm__27, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_6,
            split_7,
        ) = split_39

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_6, parameter_112, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_3 = [conv2d_12, split_7]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_111, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__117,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_13,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__117)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            relu_3, parameter_106, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full(
            [],
            float("0.982353"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_21 = paddle._C_ops.add(full_3, uniform_2)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_2 = paddle._C_ops.floor(add_21)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_2 = paddle._C_ops.divide(conv2d_14, full_3)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(divide_2, floor_2)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_3 = paddle._C_ops.add(batch_norm__27, multiply_2)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_40 = paddle._C_ops.split(add_3, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_8,
            split_9,
        ) = split_40

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_8, parameter_105, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_4 = [conv2d_15, split_9]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_104, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__118,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_16,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__118)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_4, parameter_99, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_4 = paddle._C_ops.full(
            [],
            float("0.976471"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_22 = paddle._C_ops.add(full_4, uniform_3)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_3 = paddle._C_ops.floor(add_22)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_3 = paddle._C_ops.divide(conv2d_17, full_4)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_3 = paddle._C_ops.multiply(divide_3, floor_3)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_4 = paddle._C_ops.add(add_3, multiply_3)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_41 = paddle._C_ops.split(add_4, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_41

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_10, parameter_98, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_5 = [conv2d_18, split_11]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_97, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__119,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_19,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__119)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_5, parameter_92, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [],
            float("0.970588"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_4 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_23 = paddle._C_ops.add(full_5, uniform_4)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_4 = paddle._C_ops.floor(add_23)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_4 = paddle._C_ops.divide(conv2d_20, full_5)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_4 = paddle._C_ops.multiply(divide_4, floor_4)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, multiply_4)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(add_5, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_12,
            split_13,
        ) = split_42

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_12, parameter_91, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_6 = [conv2d_21, split_13]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__120,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_22,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__120)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            relu_6, parameter_85, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_6 = paddle._C_ops.full(
            [],
            float("0.964706"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_5 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_24 = paddle._C_ops.add(full_6, uniform_5)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_5 = paddle._C_ops.floor(add_24)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_5 = paddle._C_ops.divide(conv2d_23, full_6)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_5 = paddle._C_ops.multiply(divide_5, floor_5)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, multiply_5)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_43 = paddle._C_ops.split(add_6, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_14,
            split_15,
        ) = split_43

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_14, parameter_84, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_7 = [conv2d_24, split_15]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__121,
            batch_norm__53,
            batch_norm__54,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_25,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__121)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            relu_7, parameter_78, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [],
            float("0.958824"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_6 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_25 = paddle._C_ops.add(full_7, uniform_6)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_6 = paddle._C_ops.floor(add_25)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_6 = paddle._C_ops.divide(conv2d_26, full_7)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_6 = paddle._C_ops.multiply(divide_6, floor_6)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, multiply_6)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_44 = paddle._C_ops.split(add_7, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_44

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_16, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_8 = [conv2d_27, split_17]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__122,
            batch_norm__58,
            batch_norm__59,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_28,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__122)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            relu_8, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [],
            float("0.952941"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_7 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_26 = paddle._C_ops.add(full_8, uniform_7)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_7 = paddle._C_ops.floor(add_26)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_7 = paddle._C_ops.divide(conv2d_29, full_8)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_7 = paddle._C_ops.multiply(divide_7, floor_7)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, multiply_7)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_45 = paddle._C_ops.split(add_8, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_18,
            split_19,
        ) = split_45

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_18, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_9 = [conv2d_30, split_19]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__123,
            batch_norm__63,
            batch_norm__64,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_31,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__123)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            relu_9, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_9 = paddle._C_ops.full(
            [],
            float("0.947059"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_8 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_27 = paddle._C_ops.add(full_9, uniform_8)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_8 = paddle._C_ops.floor(add_27)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_8 = paddle._C_ops.divide(conv2d_32, full_9)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_8 = paddle._C_ops.multiply(divide_8, floor_8)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, multiply_8)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_46 = paddle._C_ops.split(add_9, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_20,
            split_21,
        ) = split_46

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_20, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_10 = [conv2d_33, split_21]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__124,
            batch_norm__68,
            batch_norm__69,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_34,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__124)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            relu_10, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.941176"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_9 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_28 = paddle._C_ops.add(full_10, uniform_9)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_9 = paddle._C_ops.floor(add_28)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_9 = paddle._C_ops.divide(conv2d_35, full_10)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_9 = paddle._C_ops.multiply(divide_9, floor_9)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, multiply_9)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_47 = paddle._C_ops.split(add_10, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_47

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            split_22, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_11 = [conv2d_36, split_23]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            concat_11, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__125,
            batch_norm__73,
            batch_norm__74,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_37,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__125)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            relu_11, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.935294"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_10 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_29 = paddle._C_ops.add(full_11, uniform_10)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_10 = paddle._C_ops.floor(add_29)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_10 = paddle._C_ops.divide(conv2d_38, full_11)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_10 = paddle._C_ops.multiply(divide_10, floor_10)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, multiply_10)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_48 = paddle._C_ops.split(add_11, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_24,
            split_25,
        ) = split_48

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            split_24, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_12 = [conv2d_39, split_25]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            concat_12, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__126,
            batch_norm__78,
            batch_norm__79,
            batch_norm__80,
            batch_norm__81,
            batch_norm__82,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_40,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            relu_12, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_12 = paddle._C_ops.full(
            [],
            float("0.929412"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_11 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_30 = paddle._C_ops.add(full_12, uniform_11)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_11 = paddle._C_ops.floor(add_30)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_11 = paddle._C_ops.divide(conv2d_41, full_12)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_11 = paddle._C_ops.multiply(divide_11, floor_11)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_12 = paddle._C_ops.add(add_11, multiply_11)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_49 = paddle._C_ops.split(add_12, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_26,
            split_27,
        ) = split_49

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            split_26, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_13 = [conv2d_42, split_27]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            concat_13, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__127,
            batch_norm__83,
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_43,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__127)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(
            relu_13, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [],
            float("0.923529"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_12 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_31 = paddle._C_ops.add(full_13, uniform_12)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_12 = paddle._C_ops.floor(add_31)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_12 = paddle._C_ops.divide(conv2d_44, full_13)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_12 = paddle._C_ops.multiply(divide_12, floor_12)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_13 = paddle._C_ops.add(add_12, multiply_12)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_50 = paddle._C_ops.split(add_13, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_50

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(
            split_28, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_14 = [conv2d_45, split_29]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(
            concat_14, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__128,
            batch_norm__88,
            batch_norm__89,
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_46,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__128)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(
            relu_14, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [],
            float("0.917647"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_13 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_32 = paddle._C_ops.add(full_14, uniform_13)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_13 = paddle._C_ops.floor(add_32)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_13 = paddle._C_ops.divide(conv2d_47, full_14)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_13 = paddle._C_ops.multiply(divide_13, floor_13)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_14 = paddle._C_ops.add(add_13, multiply_13)

        # pd_op.split: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x512x14x14xf32, 2xi64, 1xi32)
        split_51 = paddle._C_ops.split(add_14, full_int_array_4, full_0)

        # builtin.split: (128x128x14x14xf32, 128x384x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32])
        (
            split_30,
            split_31,
        ) = split_51

        # pd_op.conv2d: (128x128x14x14xf32) <- (128x128x14x14xf32, 128x128x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(
            split_30, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x128x14x14xf32, 128x384x14x14xf32]) <- (128x128x14x14xf32, 128x384x14x14xf32)
        combine_15 = [conv2d_48, split_31]

        # pd_op.concat: (128x512x14x14xf32) <- ([128x128x14x14xf32, 128x384x14x14xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_0)

        # pd_op.conv2d: (128x1024x14x14xf32) <- (128x512x14x14xf32, 1024x512x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(
            concat_15, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__129,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
            batch_norm__96,
            batch_norm__97,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_49,
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

        # pd_op.relu: (128x1024x14x14xf32) <- (128x1024x14x14xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__129)

        # pd_op.conv2d: (128x512x14x14xf32) <- (128x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(
            relu_15, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_15 = paddle._C_ops.full(
            [],
            float("0.911765"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_14 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_33 = paddle._C_ops.add(full_15, uniform_14)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_14 = paddle._C_ops.floor(add_33)

        # pd_op.divide: (128x512x14x14xf32) <- (128x512x14x14xf32, xf32)
        divide_14 = paddle._C_ops.divide(conv2d_50, full_15)

        # pd_op.multiply: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x1x1x1xf32)
        multiply_14 = paddle._C_ops.multiply(divide_14, floor_14)

        # pd_op.add: (128x512x14x14xf32) <- (128x512x14x14xf32, 128x512x14x14xf32)
        add_15 = paddle._C_ops.add(add_14, multiply_14)

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x512x14x14xf32, 1024x512x2x2xf32)
        conv2d_51 = paddle._C_ops.conv2d(
            add_15, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, -1xui8) <- (128x1024x7x7xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        (
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_51,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [256, 768]

        # pd_op.split: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x1024x7x7xf32, 2xi64, 1xi32)
        split_52 = paddle._C_ops.split(batch_norm__98, full_int_array_5, full_0)

        # builtin.split: (128x256x7x7xf32, 128x768x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32])
        (
            split_32,
            split_33,
        ) = split_52

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x256x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(
            split_32, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x256x7x7xf32, 128x768x7x7xf32)
        combine_16 = [conv2d_52, split_33]

        # pd_op.concat: (128x1024x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_0)

        # pd_op.conv2d: (128x2048x7x7xf32) <- (128x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(
            concat_16, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        (
            batch_norm__130,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_53,
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

        # pd_op.relu: (128x2048x7x7xf32) <- (128x2048x7x7xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__130)

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(
            relu_16, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_16 = paddle._C_ops.full(
            [],
            float("0.905882"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_15 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_34 = paddle._C_ops.add(full_16, uniform_15)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_15 = paddle._C_ops.floor(add_34)

        # pd_op.divide: (128x1024x7x7xf32) <- (128x1024x7x7xf32, xf32)
        divide_15 = paddle._C_ops.divide(conv2d_54, full_16)

        # pd_op.multiply: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1x1x1xf32)
        multiply_15 = paddle._C_ops.multiply(divide_15, floor_15)

        # pd_op.add: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1024x7x7xf32)
        add_16 = paddle._C_ops.add(batch_norm__98, multiply_15)

        # pd_op.split: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x1024x7x7xf32, 2xi64, 1xi32)
        split_53 = paddle._C_ops.split(add_16, full_int_array_5, full_0)

        # builtin.split: (128x256x7x7xf32, 128x768x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32])
        (
            split_34,
            split_35,
        ) = split_53

        # pd_op.conv2d: (128x256x7x7xf32) <- (128x256x7x7xf32, 256x256x3x3xf32)
        conv2d_55 = paddle._C_ops.conv2d(
            split_34, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # builtin.combine: ([128x256x7x7xf32, 128x768x7x7xf32]) <- (128x256x7x7xf32, 128x768x7x7xf32)
        combine_17 = [conv2d_55, split_35]

        # pd_op.concat: (128x1024x7x7xf32) <- ([128x256x7x7xf32, 128x768x7x7xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_0)

        # pd_op.conv2d: (128x2048x7x7xf32) <- (128x1024x7x7xf32, 2048x1024x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(
            concat_17, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.batch_norm_: (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, -1xui8) <- (128x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        (
            batch_norm__131,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_56,
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

        # pd_op.relu: (128x2048x7x7xf32) <- (128x2048x7x7xf32)
        relu_17 = paddle._C_ops.relu(batch_norm__131)

        # pd_op.conv2d: (128x1024x7x7xf32) <- (128x2048x7x7xf32, 1024x2048x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(
            relu_17, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full: (xf32) <- ()
        full_17 = paddle._C_ops.full(
            [], float("0.9"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.uniform: (128x1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_16 = paddle._C_ops.uniform(
            full_int_array_3,
            paddle.float32,
            full_18,
            full_19,
            0,
            paddle.framework._current_expected_place(),
        )

        # pd_op.add: (128x1x1x1xf32) <- (xf32, 128x1x1x1xf32)
        add_35 = paddle._C_ops.add(full_17, uniform_16)

        # pd_op.floor: (128x1x1x1xf32) <- (128x1x1x1xf32)
        floor_16 = paddle._C_ops.floor(add_35)

        # pd_op.divide: (128x1024x7x7xf32) <- (128x1024x7x7xf32, xf32)
        divide_16 = paddle._C_ops.divide(conv2d_57, full_17)

        # pd_op.multiply: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1x1x1xf32)
        multiply_16 = paddle._C_ops.multiply(divide_16, floor_16)

        # pd_op.add: (128x1024x7x7xf32) <- (128x1024x7x7xf32, 128x1024x7x7xf32)
        add_17 = paddle._C_ops.add(add_16, multiply_16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.pool2d: (128x1024x1x1xf32) <- (128x1024x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_17,
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

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x1024x1x1xf32, 1280x1024x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.relu: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        relu_18 = paddle._C_ops.relu(conv2d_58)

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(relu_18, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)

        # pd_op.add: (128x102xf32) <- (128x102xf32, 102xf32)
        add_18 = paddle._C_ops.add(matmul_0, parameter_0)
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
            conv2d_4,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            assign_1,
            split_2,
            split_3,
            conv2d_5,
            assign_2,
            concat_1,
            conv2d_6,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            batch_norm__20,
            batch_norm__21,
            relu_1,
            conv2d_7,
            full_1,
            floor_0,
            divide_0,
            multiply_0,
            add_1,
            assign_3,
            split_4,
            split_5,
            conv2d_8,
            assign_4,
            concat_2,
            conv2d_9,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            batch_norm__25,
            batch_norm__26,
            relu_2,
            conv2d_10,
            full_2,
            floor_1,
            divide_1,
            multiply_1,
            add_2,
            conv2d_11,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            assign_5,
            split_6,
            split_7,
            conv2d_12,
            assign_6,
            concat_3,
            conv2d_13,
            batch_norm__33,
            batch_norm__34,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            relu_3,
            conv2d_14,
            full_3,
            floor_2,
            divide_2,
            multiply_2,
            add_3,
            assign_7,
            split_8,
            split_9,
            conv2d_15,
            assign_8,
            concat_4,
            conv2d_16,
            batch_norm__38,
            batch_norm__39,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            relu_4,
            conv2d_17,
            full_4,
            floor_3,
            divide_3,
            multiply_3,
            add_4,
            assign_9,
            split_10,
            split_11,
            conv2d_18,
            assign_10,
            concat_5,
            conv2d_19,
            batch_norm__43,
            batch_norm__44,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            relu_5,
            conv2d_20,
            full_5,
            floor_4,
            divide_4,
            multiply_4,
            add_5,
            assign_11,
            split_12,
            split_13,
            conv2d_21,
            assign_12,
            concat_6,
            conv2d_22,
            batch_norm__48,
            batch_norm__49,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            relu_6,
            conv2d_23,
            full_6,
            floor_5,
            divide_5,
            multiply_5,
            add_6,
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
            conv2d_51,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
            batch_norm__102,
            batch_norm__103,
            assign_31,
            split_32,
            split_33,
            conv2d_52,
            assign_32,
            concat_16,
            conv2d_53,
            batch_norm__104,
            batch_norm__105,
            batch_norm__106,
            batch_norm__107,
            batch_norm__108,
            relu_16,
            conv2d_54,
            full_16,
            floor_15,
            divide_15,
            multiply_15,
            add_16,
            assign_33,
            split_34,
            split_35,
            conv2d_55,
            assign_34,
            concat_17,
            conv2d_56,
            batch_norm__109,
            batch_norm__110,
            batch_norm__111,
            batch_norm__112,
            batch_norm__113,
            relu_17,
            conv2d_57,
            full_17,
            floor_16,
            divide_16,
            multiply_16,
            add_17,
            full_int_array_0,
            pool2d_0,
            relu_18,
            flatten_0,
            matmul_0,
            add_18,
        )
