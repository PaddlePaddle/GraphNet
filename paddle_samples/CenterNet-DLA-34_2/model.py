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
    ):
        # pd_op.conv2d: (4x27x16x16xf32) <- (4x512x16x16xf32, 27x512x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_3, parameter_115, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(parameter_114, full_int_array_1)

        # pd_op.add: (4x27x16x16xf32) <- (4x27x16x16xf32, 1x27x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [18, 9]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.split: ([4x18x16x16xf32, 4x9x16x16xf32]) <- (4x27x16x16xf32, 2xi64, 1xi32)
        split_16 = paddle._C_ops.split(add_28, full_int_array_2, full_0)

        # builtin.split: (4x18x16x16xf32, 4x9x16x16xf32) <- ([4x18x16x16xf32, 4x9x16x16xf32])
        (
            split_0,
            split_17,
        ) = split_16

        # pd_op.sigmoid: (4x9x16x16xf32) <- (4x9x16x16xf32)
        sigmoid_0 = paddle._C_ops.sigmoid(split_17)

        # pd_op.deformable_conv: (4x256x16x16xf32) <- (4x512x16x16xf32, 4x18x16x16xf32, 256x512x3x3xf32, 4x9x16x16xf32)
        deformable_conv_0 = paddle._C_ops.deformable_conv(
            data_3, split_0, data_4, sigmoid_0, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(data_5, full_int_array_3)

        # pd_op.add: (4x256x16x16xf32) <- (4x256x16x16xf32, 1x256x1x1xf32)
        add_0 = paddle._C_ops.add(deformable_conv_0, reshape_1)

        # pd_op.batch_norm_: (4x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__80,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_0,
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

        # pd_op.relu: (4x256x16x16xf32) <- (4x256x16x16xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__80)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_33 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_32 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_31 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_30 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_23 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_20 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_14 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_11 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_8 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_2 = full_int_array_0

        # pd_op.depthwise_conv2d_transpose: (4x256x32x32xf32) <- (4x256x16x16xf32, 256x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_0 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_0,
            parameter_109,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            256,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x256x32x32xf32) <- (4x256x32x32xf32, 4x256x32x32xf32)
        add_1 = paddle._C_ops.add(depthwise_conv2d_transpose_0, data_2)

        # pd_op.conv2d: (4x27x32x32xf32) <- (4x256x32x32xf32, 27x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            add_1, parameter_108, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(parameter_107, full_int_array_1)

        # pd_op.add: (4x27x32x32xf32) <- (4x27x32x32xf32, 1x27x1x1xf32)
        add_29 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.split: ([4x18x32x32xf32, 4x9x32x32xf32]) <- (4x27x32x32xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_29, full_int_array_2, full_0)

        # builtin.split: (4x18x32x32xf32, 4x9x32x32xf32) <- ([4x18x32x32xf32, 4x9x32x32xf32])
        (
            split_1,
            split_19,
        ) = split_18

        # pd_op.sigmoid: (4x9x32x32xf32) <- (4x9x32x32xf32)
        sigmoid_1 = paddle._C_ops.sigmoid(split_19)

        # pd_op.deformable_conv: (4x256x32x32xf32) <- (4x256x32x32xf32, 4x18x32x32xf32, 256x256x3x3xf32, 4x9x32x32xf32)
        deformable_conv_1 = paddle._C_ops.deformable_conv(
            add_1, split_1, data_6, sigmoid_1, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(data_7, full_int_array_3)

        # pd_op.add: (4x256x32x32xf32) <- (4x256x32x32xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(deformable_conv_1, reshape_3)

        # pd_op.batch_norm_: (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, -1xui8) <- (4x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        (
            batch_norm__81,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_2,
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

        # pd_op.relu: (4x256x32x32xf32) <- (4x256x32x32xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__81)

        # pd_op.conv2d: (4x27x32x32xf32) <- (4x256x32x32xf32, 27x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            data_2, parameter_102, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(parameter_101, full_int_array_1)

        # pd_op.add: (4x27x32x32xf32) <- (4x27x32x32xf32, 1x27x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_2, reshape_4)

        # pd_op.split: ([4x18x32x32xf32, 4x9x32x32xf32]) <- (4x27x32x32xf32, 2xi64, 1xi32)
        split_20 = paddle._C_ops.split(add_30, full_int_array_2, full_0)

        # builtin.split: (4x18x32x32xf32, 4x9x32x32xf32) <- ([4x18x32x32xf32, 4x9x32x32xf32])
        (
            split_2,
            split_21,
        ) = split_20

        # pd_op.sigmoid: (4x9x32x32xf32) <- (4x9x32x32xf32)
        sigmoid_2 = paddle._C_ops.sigmoid(split_21)

        # pd_op.deformable_conv: (4x128x32x32xf32) <- (4x256x32x32xf32, 4x18x32x32xf32, 128x256x3x3xf32, 4x9x32x32xf32)
        deformable_conv_2 = paddle._C_ops.deformable_conv(
            data_2, split_2, data_8, sigmoid_2, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_5 = paddle._C_ops.reshape(data_9, full_int_array_4)

        # pd_op.add: (4x128x32x32xf32) <- (4x128x32x32xf32, 1x128x1x1xf32)
        add_3 = paddle._C_ops.add(deformable_conv_2, reshape_5)

        # pd_op.batch_norm_: (4x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__82,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_3,
                parameter_100,
                parameter_99,
                parameter_98,
                parameter_97,
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

        # pd_op.relu: (4x128x32x32xf32) <- (4x128x32x32xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__82)

        # pd_op.depthwise_conv2d_transpose: (4x128x64x64xf32) <- (4x128x32x32xf32, 128x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_1 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_2,
            parameter_96,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x128x64x64xf32)
        add_4 = paddle._C_ops.add(depthwise_conv2d_transpose_1, data_1)

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            add_4, parameter_95, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(parameter_94, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_3, reshape_6)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_22 = paddle._C_ops.split(add_31, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_3,
            split_23,
        ) = split_22

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_3 = paddle._C_ops.sigmoid(split_23)

        # pd_op.deformable_conv: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 128x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_3 = paddle._C_ops.deformable_conv(
            add_4, split_3, data_10, sigmoid_3, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(data_11, full_int_array_4)

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 1x128x1x1xf32)
        add_5 = paddle._C_ops.add(deformable_conv_3, reshape_7)

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__83,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_5,
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

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__83)

        # pd_op.conv2d: (4x27x32x32xf32) <- (4x256x32x32xf32, 27x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            relu_1, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(parameter_88, full_int_array_1)

        # pd_op.add: (4x27x32x32xf32) <- (4x27x32x32xf32, 1x27x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_4, reshape_8)

        # pd_op.split: ([4x18x32x32xf32, 4x9x32x32xf32]) <- (4x27x32x32xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_32, full_int_array_2, full_0)

        # builtin.split: (4x18x32x32xf32, 4x9x32x32xf32) <- ([4x18x32x32xf32, 4x9x32x32xf32])
        (
            split_4,
            split_25,
        ) = split_24

        # pd_op.sigmoid: (4x9x32x32xf32) <- (4x9x32x32xf32)
        sigmoid_4 = paddle._C_ops.sigmoid(split_25)

        # pd_op.deformable_conv: (4x128x32x32xf32) <- (4x256x32x32xf32, 4x18x32x32xf32, 128x256x3x3xf32, 4x9x32x32xf32)
        deformable_conv_4 = paddle._C_ops.deformable_conv(
            relu_1, split_4, data_12, sigmoid_4, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(data_13, full_int_array_4)

        # pd_op.add: (4x128x32x32xf32) <- (4x128x32x32xf32, 1x128x1x1xf32)
        add_6 = paddle._C_ops.add(deformable_conv_4, reshape_9)

        # pd_op.batch_norm_: (4x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__84,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_6,
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

        # pd_op.relu: (4x128x32x32xf32) <- (4x128x32x32xf32)
        relu_4 = paddle._C_ops.relu(batch_norm__84)

        # pd_op.depthwise_conv2d_transpose: (4x128x64x64xf32) <- (4x128x32x32xf32, 128x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_2 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_4,
            parameter_83,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            128,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x128x64x64xf32)
        add_7 = paddle._C_ops.add(depthwise_conv2d_transpose_2, relu_3)

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            add_7, parameter_82, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(parameter_81, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_33 = paddle._C_ops.add(conv2d_5, reshape_10)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_26 = paddle._C_ops.split(add_33, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_5,
            split_27,
        ) = split_26

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_5 = paddle._C_ops.sigmoid(split_27)

        # pd_op.deformable_conv: (4x128x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 128x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_5 = paddle._C_ops.deformable_conv(
            add_7, split_5, data_14, sigmoid_5, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x128x1x1xf32) <- (128xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(data_15, full_int_array_4)

        # pd_op.add: (4x128x64x64xf32) <- (4x128x64x64xf32, 1x128x1x1xf32)
        add_8 = paddle._C_ops.add(deformable_conv_5, reshape_11)

        # pd_op.batch_norm_: (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, -1xui8) <- (4x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        (
            batch_norm__85,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_8,
                parameter_80,
                parameter_79,
                parameter_78,
                parameter_77,
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

        # pd_op.relu: (4x128x64x64xf32) <- (4x128x64x64xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__85)

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            data_1, parameter_76, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(parameter_75, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_34 = paddle._C_ops.add(conv2d_6, reshape_12)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_28 = paddle._C_ops.split(add_34, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_6,
            split_29,
        ) = split_28

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_6 = paddle._C_ops.sigmoid(split_29)

        # pd_op.deformable_conv: (4x64x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 64x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_6 = paddle._C_ops.deformable_conv(
            data_1, split_6, data_16, sigmoid_6, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(data_17, full_int_array_5)

        # pd_op.add: (4x64x64x64xf32) <- (4x64x64x64xf32, 1x64x1x1xf32)
        add_9 = paddle._C_ops.add(deformable_conv_6, reshape_13)

        # pd_op.batch_norm_: (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__86,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_9,
                parameter_74,
                parameter_73,
                parameter_72,
                parameter_71,
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

        # pd_op.relu: (4x64x64x64xf32) <- (4x64x64x64xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__86)

        # pd_op.depthwise_conv2d_transpose: (4x64x128x128xf32) <- (4x64x64x64xf32, 64x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_3 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_6,
            parameter_70,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x64x128x128xf32)
        add_10 = paddle._C_ops.add(depthwise_conv2d_transpose_3, data_0)

        # pd_op.conv2d: (4x27x128x128xf32) <- (4x64x128x128xf32, 27x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            add_10, parameter_69, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_14 = paddle._C_ops.reshape(parameter_68, full_int_array_1)

        # pd_op.add: (4x27x128x128xf32) <- (4x27x128x128xf32, 1x27x1x1xf32)
        add_35 = paddle._C_ops.add(conv2d_7, reshape_14)

        # pd_op.split: ([4x18x128x128xf32, 4x9x128x128xf32]) <- (4x27x128x128xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_35, full_int_array_2, full_0)

        # builtin.split: (4x18x128x128xf32, 4x9x128x128xf32) <- ([4x18x128x128xf32, 4x9x128x128xf32])
        (
            split_7,
            split_31,
        ) = split_30

        # pd_op.sigmoid: (4x9x128x128xf32) <- (4x9x128x128xf32)
        sigmoid_7 = paddle._C_ops.sigmoid(split_31)

        # pd_op.deformable_conv: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x18x128x128xf32, 64x64x3x3xf32, 4x9x128x128xf32)
        deformable_conv_7 = paddle._C_ops.deformable_conv(
            add_10, split_7, data_18, sigmoid_7, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_15 = paddle._C_ops.reshape(data_19, full_int_array_5)

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 1x64x1x1xf32)
        add_11 = paddle._C_ops.add(deformable_conv_7, reshape_15)

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__87,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_11,
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

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        relu_7 = paddle._C_ops.relu(batch_norm__87)

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            relu_3, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_16 = paddle._C_ops.reshape(parameter_62, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_36 = paddle._C_ops.add(conv2d_8, reshape_16)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_32 = paddle._C_ops.split(add_36, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_8,
            split_33,
        ) = split_32

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_8 = paddle._C_ops.sigmoid(split_33)

        # pd_op.deformable_conv: (4x64x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 64x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_8 = paddle._C_ops.deformable_conv(
            relu_3, split_8, data_20, sigmoid_8, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_17 = paddle._C_ops.reshape(data_21, full_int_array_5)

        # pd_op.add: (4x64x64x64xf32) <- (4x64x64x64xf32, 1x64x1x1xf32)
        add_12 = paddle._C_ops.add(deformable_conv_8, reshape_17)

        # pd_op.batch_norm_: (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__88,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_12,
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

        # pd_op.relu: (4x64x64x64xf32) <- (4x64x64x64xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__88)

        # pd_op.depthwise_conv2d_transpose: (4x64x128x128xf32) <- (4x64x64x64xf32, 64x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_4 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_8,
            parameter_57,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x64x128x128xf32)
        add_13 = paddle._C_ops.add(depthwise_conv2d_transpose_4, relu_7)

        # pd_op.conv2d: (4x27x128x128xf32) <- (4x64x128x128xf32, 27x64x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            add_13, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_18 = paddle._C_ops.reshape(parameter_55, full_int_array_1)

        # pd_op.add: (4x27x128x128xf32) <- (4x27x128x128xf32, 1x27x1x1xf32)
        add_37 = paddle._C_ops.add(conv2d_9, reshape_18)

        # pd_op.split: ([4x18x128x128xf32, 4x9x128x128xf32]) <- (4x27x128x128xf32, 2xi64, 1xi32)
        split_34 = paddle._C_ops.split(add_37, full_int_array_2, full_0)

        # builtin.split: (4x18x128x128xf32, 4x9x128x128xf32) <- ([4x18x128x128xf32, 4x9x128x128xf32])
        (
            split_9,
            split_35,
        ) = split_34

        # pd_op.sigmoid: (4x9x128x128xf32) <- (4x9x128x128xf32)
        sigmoid_9 = paddle._C_ops.sigmoid(split_35)

        # pd_op.deformable_conv: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x18x128x128xf32, 64x64x3x3xf32, 4x9x128x128xf32)
        deformable_conv_9 = paddle._C_ops.deformable_conv(
            add_13, split_9, data_22, sigmoid_9, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_19 = paddle._C_ops.reshape(data_23, full_int_array_5)

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 1x64x1x1xf32)
        add_14 = paddle._C_ops.add(deformable_conv_9, reshape_19)

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__89,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_14,
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

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__89)

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            relu_5, parameter_50, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_20 = paddle._C_ops.reshape(parameter_49, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_38 = paddle._C_ops.add(conv2d_10, reshape_20)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_38, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_10,
            split_37,
        ) = split_36

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_10 = paddle._C_ops.sigmoid(split_37)

        # pd_op.deformable_conv: (4x64x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 64x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_10 = paddle._C_ops.deformable_conv(
            relu_5, split_10, data_24, sigmoid_10, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_21 = paddle._C_ops.reshape(data_25, full_int_array_5)

        # pd_op.add: (4x64x64x64xf32) <- (4x64x64x64xf32, 1x64x1x1xf32)
        add_15 = paddle._C_ops.add(deformable_conv_10, reshape_21)

        # pd_op.batch_norm_: (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__90,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_15,
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

        # pd_op.relu: (4x64x64x64xf32) <- (4x64x64x64xf32)
        relu_10 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.depthwise_conv2d_transpose: (4x64x128x128xf32) <- (4x64x64x64xf32, 64x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_5 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_10,
            parameter_44,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x64x128x128xf32)
        add_16 = paddle._C_ops.add(depthwise_conv2d_transpose_5, relu_9)

        # pd_op.conv2d: (4x27x128x128xf32) <- (4x64x128x128xf32, 27x64x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_16, parameter_43, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_22 = paddle._C_ops.reshape(parameter_42, full_int_array_1)

        # pd_op.add: (4x27x128x128xf32) <- (4x27x128x128xf32, 1x27x1x1xf32)
        add_39 = paddle._C_ops.add(conv2d_11, reshape_22)

        # pd_op.split: ([4x18x128x128xf32, 4x9x128x128xf32]) <- (4x27x128x128xf32, 2xi64, 1xi32)
        split_38 = paddle._C_ops.split(add_39, full_int_array_2, full_0)

        # builtin.split: (4x18x128x128xf32, 4x9x128x128xf32) <- ([4x18x128x128xf32, 4x9x128x128xf32])
        (
            split_11,
            split_39,
        ) = split_38

        # pd_op.sigmoid: (4x9x128x128xf32) <- (4x9x128x128xf32)
        sigmoid_11 = paddle._C_ops.sigmoid(split_39)

        # pd_op.deformable_conv: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x18x128x128xf32, 64x64x3x3xf32, 4x9x128x128xf32)
        deformable_conv_11 = paddle._C_ops.deformable_conv(
            add_16, split_11, data_26, sigmoid_11, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_23 = paddle._C_ops.reshape(data_27, full_int_array_5)

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 1x64x1x1xf32)
        add_17 = paddle._C_ops.add(deformable_conv_11, reshape_23)

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__91,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_17,
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

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__91)

        # pd_op.assign: (4x64x128x128xf32) <- (4x64x128x128xf32)
        assign_16 = relu_11

        # pd_op.assign: (4x128x64x64xf32) <- (4x128x64x64xf32)
        assign_17 = relu_5

        # pd_op.assign: (4x256x32x32xf32) <- (4x256x32x32xf32)
        assign_18 = relu_1

        # pd_op.conv2d: (4x27x64x64xf32) <- (4x128x64x64xf32, 27x128x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            assign_17, parameter_37, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_24 = paddle._C_ops.reshape(parameter_36, full_int_array_1)

        # pd_op.add: (4x27x64x64xf32) <- (4x27x64x64xf32, 1x27x1x1xf32)
        add_40 = paddle._C_ops.add(conv2d_12, reshape_24)

        # pd_op.split: ([4x18x64x64xf32, 4x9x64x64xf32]) <- (4x27x64x64xf32, 2xi64, 1xi32)
        split_40 = paddle._C_ops.split(add_40, full_int_array_2, full_0)

        # builtin.split: (4x18x64x64xf32, 4x9x64x64xf32) <- ([4x18x64x64xf32, 4x9x64x64xf32])
        (
            split_12,
            split_41,
        ) = split_40

        # pd_op.sigmoid: (4x9x64x64xf32) <- (4x9x64x64xf32)
        sigmoid_12 = paddle._C_ops.sigmoid(split_41)

        # pd_op.deformable_conv: (4x64x64x64xf32) <- (4x128x64x64xf32, 4x18x64x64xf32, 64x128x3x3xf32, 4x9x64x64xf32)
        deformable_conv_12 = paddle._C_ops.deformable_conv(
            assign_17, split_12, data_28, sigmoid_12, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_25 = paddle._C_ops.reshape(data_29, full_int_array_5)

        # pd_op.add: (4x64x64x64xf32) <- (4x64x64x64xf32, 1x64x1x1xf32)
        add_18 = paddle._C_ops.add(deformable_conv_12, reshape_25)

        # pd_op.batch_norm_: (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__92,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_18,
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

        # pd_op.relu: (4x64x64x64xf32) <- (4x64x64x64xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__92)

        # pd_op.depthwise_conv2d_transpose: (4x64x128x128xf32) <- (4x64x64x64xf32, 64x1x4x4xf32, 0xi64)
        depthwise_conv2d_transpose_6 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_12,
            parameter_31,
            [2, 2],
            [1, 1],
            [],
            full_int_array_0,
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x64x128x128xf32)
        add_19 = paddle._C_ops.add(depthwise_conv2d_transpose_6, assign_16)

        # pd_op.conv2d: (4x27x128x128xf32) <- (4x64x128x128xf32, 27x64x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            add_19, parameter_30, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_26 = paddle._C_ops.reshape(parameter_29, full_int_array_1)

        # pd_op.add: (4x27x128x128xf32) <- (4x27x128x128xf32, 1x27x1x1xf32)
        add_41 = paddle._C_ops.add(conv2d_13, reshape_26)

        # pd_op.split: ([4x18x128x128xf32, 4x9x128x128xf32]) <- (4x27x128x128xf32, 2xi64, 1xi32)
        split_42 = paddle._C_ops.split(add_41, full_int_array_2, full_0)

        # builtin.split: (4x18x128x128xf32, 4x9x128x128xf32) <- ([4x18x128x128xf32, 4x9x128x128xf32])
        (
            split_13,
            split_43,
        ) = split_42

        # pd_op.sigmoid: (4x9x128x128xf32) <- (4x9x128x128xf32)
        sigmoid_13 = paddle._C_ops.sigmoid(split_43)

        # pd_op.deformable_conv: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x18x128x128xf32, 64x64x3x3xf32, 4x9x128x128xf32)
        deformable_conv_13 = paddle._C_ops.deformable_conv(
            add_19, split_13, data_30, sigmoid_13, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_27 = paddle._C_ops.reshape(data_31, full_int_array_5)

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 1x64x1x1xf32)
        add_20 = paddle._C_ops.add(deformable_conv_13, reshape_27)

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__93,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_20,
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

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        relu_13 = paddle._C_ops.relu(batch_norm__93)

        # pd_op.conv2d: (4x27x32x32xf32) <- (4x256x32x32xf32, 27x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            assign_18, parameter_24, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_28 = paddle._C_ops.reshape(parameter_23, full_int_array_1)

        # pd_op.add: (4x27x32x32xf32) <- (4x27x32x32xf32, 1x27x1x1xf32)
        add_42 = paddle._C_ops.add(conv2d_14, reshape_28)

        # pd_op.split: ([4x18x32x32xf32, 4x9x32x32xf32]) <- (4x27x32x32xf32, 2xi64, 1xi32)
        split_44 = paddle._C_ops.split(add_42, full_int_array_2, full_0)

        # builtin.split: (4x18x32x32xf32, 4x9x32x32xf32) <- ([4x18x32x32xf32, 4x9x32x32xf32])
        (
            split_14,
            split_45,
        ) = split_44

        # pd_op.sigmoid: (4x9x32x32xf32) <- (4x9x32x32xf32)
        sigmoid_14 = paddle._C_ops.sigmoid(split_45)

        # pd_op.deformable_conv: (4x64x32x32xf32) <- (4x256x32x32xf32, 4x18x32x32xf32, 64x256x3x3xf32, 4x9x32x32xf32)
        deformable_conv_14 = paddle._C_ops.deformable_conv(
            assign_18, split_14, data_32, sigmoid_14, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_29 = paddle._C_ops.reshape(data_33, full_int_array_5)

        # pd_op.add: (4x64x32x32xf32) <- (4x64x32x32xf32, 1x64x1x1xf32)
        add_21 = paddle._C_ops.add(deformable_conv_14, reshape_29)

        # pd_op.batch_norm_: (4x64x32x32xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x32x32xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__94,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_21,
                parameter_22,
                parameter_21,
                parameter_20,
                parameter_19,
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

        # pd_op.relu: (4x64x32x32xf32) <- (4x64x32x32xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__94)

        # pd_op.depthwise_conv2d_transpose: (4x64x128x128xf32) <- (4x64x32x32xf32, 64x1x8x8xf32, 0xi64)
        depthwise_conv2d_transpose_7 = paddle._C_ops.depthwise_conv2d_transpose(
            relu_14,
            parameter_18,
            [4, 4],
            [2, 2],
            [],
            full_int_array_0,
            "EXPLICIT",
            64,
            [1, 1],
            "NCHW",
        )

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x64x128x128xf32)
        add_22 = paddle._C_ops.add(depthwise_conv2d_transpose_7, relu_13)

        # pd_op.conv2d: (4x27x128x128xf32) <- (4x64x128x128xf32, 27x64x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            add_22, parameter_17, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x27x1x1xf32) <- (27xf32, 4xi64)
        reshape_30 = paddle._C_ops.reshape(parameter_16, full_int_array_1)

        # pd_op.add: (4x27x128x128xf32) <- (4x27x128x128xf32, 1x27x1x1xf32)
        add_43 = paddle._C_ops.add(conv2d_15, reshape_30)

        # pd_op.split: ([4x18x128x128xf32, 4x9x128x128xf32]) <- (4x27x128x128xf32, 2xi64, 1xi32)
        split_46 = paddle._C_ops.split(add_43, full_int_array_2, full_0)

        # builtin.split: (4x18x128x128xf32, 4x9x128x128xf32) <- ([4x18x128x128xf32, 4x9x128x128xf32])
        (
            split_15,
            split_47,
        ) = split_46

        # pd_op.sigmoid: (4x9x128x128xf32) <- (4x9x128x128xf32)
        sigmoid_15 = paddle._C_ops.sigmoid(split_47)

        # pd_op.deformable_conv: (4x64x128x128xf32) <- (4x64x128x128xf32, 4x18x128x128xf32, 64x64x3x3xf32, 4x9x128x128xf32)
        deformable_conv_15 = paddle._C_ops.deformable_conv(
            add_22, split_15, data_34, sigmoid_15, [1, 1], [1, 1], [1, 1], 1, 1, 1
        )

        # pd_op.reshape: (1x64x1x1xf32) <- (64xf32, 4xi64)
        reshape_31 = paddle._C_ops.reshape(data_35, full_int_array_5)

        # pd_op.add: (4x64x128x128xf32) <- (4x64x128x128xf32, 1x64x1x1xf32)
        add_23 = paddle._C_ops.add(deformable_conv_15, reshape_31)

        # pd_op.batch_norm_: (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, -1xui8) <- (4x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        (
            batch_norm__95,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                add_23,
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

        # pd_op.relu: (4x64x128x128xf32) <- (4x64x128x128xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__95)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            relu_15, parameter_11, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_32 = paddle._C_ops.reshape(parameter_10, full_int_array_1)

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        add_44 = paddle._C_ops.add(conv2d_16, reshape_32)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        relu_16 = paddle._C_ops.relu(add_44)

        # pd_op.conv2d: (4x4x128x128xf32) <- (4x256x128x128xf32, 4x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            relu_16, parameter_9, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x4x1x1xf32) <- (4xf32, 4xi64)
        reshape_33 = paddle._C_ops.reshape(parameter_8, full_int_array_1)

        # pd_op.add: (4x4x128x128xf32) <- (4x4x128x128xf32, 1x4x1x1xf32)
        add_45 = paddle._C_ops.add(conv2d_17, reshape_33)

        # pd_op.sigmoid: (4x4x128x128xf32) <- (4x4x128x128xf32)
        sigmoid_16 = paddle._C_ops.sigmoid(add_45)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            relu_15, parameter_7, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_34 = paddle._C_ops.reshape(parameter_6, full_int_array_1)

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        add_46 = paddle._C_ops.add(conv2d_18, reshape_34)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        relu_17 = paddle._C_ops.relu(add_46)

        # pd_op.conv2d: (4x2x128x128xf32) <- (4x256x128x128xf32, 2x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            relu_17, parameter_5, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_35 = paddle._C_ops.reshape(parameter_4, full_int_array_1)

        # pd_op.add: (4x2x128x128xf32) <- (4x2x128x128xf32, 1x2x1x1xf32)
        add_47 = paddle._C_ops.add(conv2d_19, reshape_35)

        # pd_op.conv2d: (4x256x128x128xf32) <- (4x64x128x128xf32, 256x64x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            relu_15, parameter_3, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x256x1x1xf32) <- (256xf32, 4xi64)
        reshape_36 = paddle._C_ops.reshape(parameter_2, full_int_array_1)

        # pd_op.add: (4x256x128x128xf32) <- (4x256x128x128xf32, 1x256x1x1xf32)
        add_48 = paddle._C_ops.add(conv2d_20, reshape_36)

        # pd_op.relu: (4x256x128x128xf32) <- (4x256x128x128xf32)
        relu_18 = paddle._C_ops.relu(add_48)

        # pd_op.conv2d: (4x2x128x128xf32) <- (4x256x128x128xf32, 2x256x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            relu_18, parameter_1, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )

        # pd_op.reshape: (1x2x1x1xf32) <- (2xf32, 4xi64)
        reshape_37 = paddle._C_ops.reshape(parameter_0, full_int_array_1)

        # pd_op.add: (4x2x128x128xf32) <- (4x2x128x128xf32, 1x2x1x1xf32)
        add_49 = paddle._C_ops.add(conv2d_21, reshape_37)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.0001"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0.9999"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(sigmoid_16, full_1, full_2)

        # pd_op.full: (xf32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("1"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (4x4x128x128xb) <- (4x4x128x128xf32, xf32)
        equal_0 = paddle._C_ops.equal(data_38, full_5)

        # pd_op.cast: (4x4x128x128xf32) <- (4x4x128x128xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.float32)

        # pd_op.less_than: (4x4x128x128xb) <- (4x4x128x128xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(data_38, full_5)

        # pd_op.cast: (4x4x128x128xf32) <- (4x4x128x128xb)
        cast_1 = paddle._C_ops.cast(less_than_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_29 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_27 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_26 = full_6

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_25 = full_6

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(data_38, full_6, float("1"), True)

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        pow_0 = paddle._C_ops.pow(scale_9, float("4"))

        # pd_op.log: (4x4x128x128xf32) <- (4x4x128x128xf32)
        log_0 = paddle._C_ops.log(clip_0)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(clip_0, full_6, float("1"), True)

        # pd_op.assign: (4x4x128x128xf32) <- (4x4x128x128xf32)
        assign_28 = scale_0

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        pow_1 = paddle._C_ops.pow(scale_0, float("2"))

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        multiply_0 = paddle._C_ops.multiply(log_0, pow_1)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        multiply_7 = paddle._C_ops.multiply(multiply_0, cast_0)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_7, full_6, float("0"), True)

        # pd_op.log: (4x4x128x128xf32) <- (4x4x128x128xf32)
        log_1 = paddle._C_ops.log(scale_0)

        # pd_op.pow: (4x4x128x128xf32) <- (4x4x128x128xf32)
        pow_2 = paddle._C_ops.pow(clip_0, float("2"))

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        multiply_1 = paddle._C_ops.multiply(log_1, pow_2)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        multiply_2 = paddle._C_ops.multiply(multiply_1, pow_0)

        # pd_op.multiply: (4x4x128x128xf32) <- (4x4x128x128xf32, 4x4x128x128xf32)
        multiply_8 = paddle._C_ops.multiply(multiply_2, cast_1)

        # pd_op.scale: (4x4x128x128xf32) <- (4x4x128x128xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(multiply_8, full_6, float("0"), True)

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(scale_1, full_int_array_0, None, False)

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(scale_2, full_int_array_0, None, False)

        # pd_op.sum: (xf32) <- (4x4x128x128xf32, 0xi64)
        sum_4 = paddle._C_ops.sum(cast_0, full_int_array_0, None, False)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_24 = paddle._C_ops.add(sum_0, sum_1)

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xf32, xf32)
        equal_1 = paddle._C_ops.equal(sum_4, full_7)

        # pd_op.cast: (xf32) <- (xb)
        cast_4 = paddle._C_ops.cast(equal_1, paddle.float32)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_25 = paddle._C_ops.add(sum_4, cast_4)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(add_24, add_25)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_35 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_34 = full_3

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(divide_0, full_3, float("0"), True)

        # pd_op.transpose: (4x128x128x2xf32) <- (4x2x128x128xf32)
        transpose_0 = paddle._C_ops.transpose(add_47, [0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [4, -1, 2]

        # pd_op.reshape: (4x16384x2xf32) <- (4x128x128x2xf32, 3xi64)
        reshape_38 = paddle._C_ops.reshape(transpose_0, full_int_array_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.unsqueeze: (4x128x1xi64) <- (4x128xi64, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_36, full_int_array_7)

        # pd_op.full: (1x128x1xi64) <- ()
        full_8 = paddle._C_ops.full(
            [1, 128, 1],
            float("0"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        full_9 = paddle._C_ops.full(
            [1, 128, 1],
            float("1"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        full_10 = paddle._C_ops.full(
            [1, 128, 1],
            float("2"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1x128x1xi64) <- ()
        full_11 = paddle._C_ops.full(
            [1, 128, 1],
            float("3"),
            paddle.int64,
            paddle.framework._current_expected_place(),
        )

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64]) <- (1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64)
        combine_0 = [full_8, full_9, full_10, full_11]

        # pd_op.concat: (4x128x1xi64) <- ([1x128x1xi64, 1x128x1xi64, 1x128x1xi64, 1x128x1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_12)

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full(
            [1], float("2"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([4x128x1xi64, 4x128x1xi64]) <- (4x128x1xi64, 4x128x1xi64)
        combine_1 = [concat_1, unsqueeze_0]

        # pd_op.concat: (4x128x2xi64) <- ([4x128x1xi64, 4x128x1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_13)

        # pd_op.gather_nd: (4x128x2xf32) <- (4x16384x2xf32, 4x128x2xi64)
        gather_nd_0 = paddle._C_ops.gather_nd(reshape_38, concat_0)

        # pd_op.unsqueeze: (4x128x1xi32) <- (4x128xi32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(data_37, full_int_array_7)

        # pd_op.expand_as: (4x128x2xi32) <- (4x128x1xi32, 4x128x2xf32)
        expand_as_0 = paddle._C_ops.expand_as(unsqueeze_1, gather_nd_0, [4, 128, 2])

        # pd_op.cast: (4x128x2xf32) <- (4x128x2xi32)
        cast_2 = paddle._C_ops.cast(expand_as_0, paddle.float32)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        sum_5 = paddle._C_ops.sum(cast_2, full_int_array_0, None, False)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        multiply_3 = paddle._C_ops.multiply(gather_nd_0, cast_2)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        multiply_4 = paddle._C_ops.multiply(data_39, cast_2)

        # pd_op.subtract: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        subtract_0 = paddle._C_ops.subtract(multiply_3, multiply_4)

        # pd_op.abs: (4x128x2xf32) <- (4x128x2xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(abs_0, full_int_array_0, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(sum_5, full_3, float("0.0001"), True)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_1 = paddle._C_ops.divide(sum_2, scale_3)

        # pd_op.transpose: (4x128x128x2xf32) <- (4x2x128x128xf32)
        transpose_1 = paddle._C_ops.transpose(add_49, [0, 2, 3, 1])

        # pd_op.reshape: (4x16384x2xf32) <- (4x128x128x2xf32, 3xi64)
        reshape_39 = paddle._C_ops.reshape(transpose_1, full_int_array_6)

        # pd_op.gather_nd: (4x128x2xf32) <- (4x16384x2xf32, 4x128x2xi64)
        gather_nd_1 = paddle._C_ops.gather_nd(reshape_39, concat_0)

        # pd_op.expand_as: (4x128x2xi32) <- (4x128x1xi32, 4x128x2xf32)
        expand_as_1 = paddle._C_ops.expand_as(unsqueeze_1, gather_nd_1, [4, 128, 2])

        # pd_op.cast: (4x128x2xf32) <- (4x128x2xi32)
        cast_3 = paddle._C_ops.cast(expand_as_1, paddle.float32)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        sum_6 = paddle._C_ops.sum(cast_3, full_int_array_0, None, False)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        multiply_5 = paddle._C_ops.multiply(gather_nd_1, cast_3)

        # pd_op.multiply: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        multiply_6 = paddle._C_ops.multiply(data_40, cast_3)

        # pd_op.subtract: (4x128x2xf32) <- (4x128x2xf32, 4x128x2xf32)
        subtract_1 = paddle._C_ops.subtract(multiply_5, multiply_6)

        # pd_op.abs: (4x128x2xf32) <- (4x128x2xf32)
        abs_1 = paddle._C_ops.abs(subtract_1)

        # pd_op.sum: (xf32) <- (4x128x2xf32, 0xi64)
        sum_3 = paddle._C_ops.sum(abs_1, full_int_array_0, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(sum_6, full_3, float("0.0001"), True)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_2 = paddle._C_ops.divide(sum_3, scale_4)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_8, full_3, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0.1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(divide_1, full_4, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_26 = paddle._C_ops.add(scale_5, scale_6)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(divide_2, full_3, float("0"), True)

        # pd_op.add: (xf32) <- (xf32, xf32)
        add_27 = paddle._C_ops.add(add_26, scale_7)
        return (
            conv2d_0,
            reshape_0,
            full_0,
            split_0,
            sigmoid_0,
            deformable_conv_0,
            reshape_1,
            add_0,
            batch_norm__0,
            batch_norm__1,
            batch_norm__2,
            batch_norm__3,
            batch_norm__4,
            relu_0,
            full_int_array_0,
            depthwise_conv2d_transpose_0,
            add_1,
            conv2d_1,
            reshape_2,
            assign_0,
            split_1,
            sigmoid_1,
            deformable_conv_1,
            reshape_3,
            add_2,
            batch_norm__5,
            batch_norm__6,
            batch_norm__7,
            batch_norm__8,
            batch_norm__9,
            relu_1,
            conv2d_2,
            reshape_4,
            assign_1,
            split_2,
            sigmoid_2,
            deformable_conv_2,
            reshape_5,
            add_3,
            batch_norm__10,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            relu_2,
            assign_2,
            depthwise_conv2d_transpose_1,
            add_4,
            conv2d_3,
            reshape_6,
            assign_3,
            split_3,
            sigmoid_3,
            deformable_conv_3,
            reshape_7,
            add_5,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
            relu_3,
            conv2d_4,
            reshape_8,
            assign_4,
            split_4,
            sigmoid_4,
            deformable_conv_4,
            reshape_9,
            add_6,
            batch_norm__20,
            batch_norm__21,
            batch_norm__22,
            batch_norm__23,
            batch_norm__24,
            relu_4,
            assign_5,
            depthwise_conv2d_transpose_2,
            add_7,
            conv2d_5,
            reshape_10,
            assign_6,
            split_5,
            sigmoid_5,
            deformable_conv_5,
            reshape_11,
            add_8,
            batch_norm__25,
            batch_norm__26,
            batch_norm__27,
            batch_norm__28,
            batch_norm__29,
            relu_5,
            conv2d_6,
            reshape_12,
            assign_7,
            split_6,
            sigmoid_6,
            deformable_conv_6,
            reshape_13,
            add_9,
            batch_norm__30,
            batch_norm__31,
            batch_norm__32,
            batch_norm__33,
            batch_norm__34,
            relu_6,
            assign_8,
            depthwise_conv2d_transpose_3,
            add_10,
            conv2d_7,
            reshape_14,
            assign_9,
            split_7,
            sigmoid_7,
            deformable_conv_7,
            reshape_15,
            add_11,
            batch_norm__35,
            batch_norm__36,
            batch_norm__37,
            batch_norm__38,
            batch_norm__39,
            relu_7,
            conv2d_8,
            reshape_16,
            assign_10,
            split_8,
            sigmoid_8,
            deformable_conv_8,
            reshape_17,
            add_12,
            batch_norm__40,
            batch_norm__41,
            batch_norm__42,
            batch_norm__43,
            batch_norm__44,
            relu_8,
            assign_11,
            depthwise_conv2d_transpose_4,
            add_13,
            conv2d_9,
            reshape_18,
            assign_12,
            split_9,
            sigmoid_9,
            deformable_conv_9,
            reshape_19,
            add_14,
            batch_norm__45,
            batch_norm__46,
            batch_norm__47,
            batch_norm__48,
            batch_norm__49,
            relu_9,
            conv2d_10,
            reshape_20,
            assign_13,
            split_10,
            sigmoid_10,
            deformable_conv_10,
            reshape_21,
            add_15,
            batch_norm__50,
            batch_norm__51,
            batch_norm__52,
            batch_norm__53,
            batch_norm__54,
            relu_10,
            assign_14,
            depthwise_conv2d_transpose_5,
            add_16,
            conv2d_11,
            reshape_22,
            assign_15,
            split_11,
            sigmoid_11,
            deformable_conv_11,
            reshape_23,
            add_17,
            batch_norm__55,
            batch_norm__56,
            batch_norm__57,
            batch_norm__58,
            batch_norm__59,
            relu_11,
            assign_16,
            assign_17,
            assign_18,
            conv2d_12,
            reshape_24,
            assign_19,
            split_12,
            sigmoid_12,
            deformable_conv_12,
            reshape_25,
            add_18,
            batch_norm__60,
            batch_norm__61,
            batch_norm__62,
            batch_norm__63,
            batch_norm__64,
            relu_12,
            assign_20,
            depthwise_conv2d_transpose_6,
            add_19,
            conv2d_13,
            reshape_26,
            assign_21,
            split_13,
            sigmoid_13,
            deformable_conv_13,
            reshape_27,
            add_20,
            batch_norm__65,
            batch_norm__66,
            batch_norm__67,
            batch_norm__68,
            batch_norm__69,
            relu_13,
            conv2d_14,
            reshape_28,
            assign_22,
            split_14,
            sigmoid_14,
            deformable_conv_14,
            reshape_29,
            add_21,
            batch_norm__70,
            batch_norm__71,
            batch_norm__72,
            batch_norm__73,
            batch_norm__74,
            relu_14,
            assign_23,
            depthwise_conv2d_transpose_7,
            add_22,
            conv2d_15,
            reshape_30,
            assign_24,
            split_15,
            sigmoid_15,
            deformable_conv_15,
            reshape_31,
            add_23,
            batch_norm__75,
            batch_norm__76,
            batch_norm__77,
            batch_norm__78,
            batch_norm__79,
            relu_15,
            conv2d_16,
            reshape_32,
            relu_16,
            conv2d_17,
            reshape_33,
            sigmoid_16,
            conv2d_18,
            reshape_34,
            relu_17,
            conv2d_19,
            reshape_35,
            conv2d_20,
            reshape_36,
            relu_18,
            conv2d_21,
            reshape_37,
            full_1,
            full_2,
            clip_0,
            cast_0,
            cast_1,
            pow_0,
            log_0,
            assign_25,
            scale_0,
            pow_1,
            multiply_0,
            assign_26,
            scale_1,
            assign_27,
            assign_28,
            log_1,
            pow_2,
            multiply_1,
            multiply_2,
            assign_29,
            scale_2,
            assign_30,
            sum_0,
            assign_31,
            sum_1,
            add_24,
            add_25,
            divide_0,
            full_3,
            transpose_0,
            reshape_38,
            concat_0,
            gather_nd_0,
            cast_2,
            multiply_3,
            multiply_4,
            subtract_0,
            abs_0,
            assign_32,
            sum_2,
            scale_3,
            transpose_1,
            reshape_39,
            gather_nd_1,
            cast_3,
            multiply_5,
            multiply_6,
            subtract_1,
            abs_1,
            assign_33,
            sum_3,
            scale_4,
            assign_34,
            scale_5,
            full_4,
            scale_6,
            add_26,
            assign_35,
            scale_7,
            scale_8,
            divide_1,
            divide_2,
            add_27,
        )
