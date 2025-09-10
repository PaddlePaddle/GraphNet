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
        data_0,
    ):
        # pd_op.conv2d: (128x40x56x56xf32) <- (128x3x224x224xf32, 40x3x4x4xf32)
        conv2d_0 = paddle._C_ops.conv2d(
            data_0, parameter_113, [4, 4], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del data_0, parameter_113

        # pd_op.batch_norm_: (128x40x56x56xf32, 40xf32, 40xf32, 40xf32, 40xf32, -1xui8) <- (128x40x56x56xf32, 40xf32, 40xf32, 40xf32, 40xf32)
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
                parameter_112,
                parameter_111,
                parameter_110,
                parameter_109,
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
        del parameter_109, parameter_110, parameter_111, parameter_112

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [10, 30]

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

        # pd_op.split: ([128x10x56x56xf32, 128x30x56x56xf32]) <- (128x40x56x56xf32, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(batch_norm__0, full_int_array_0, full_0)
        del full_int_array_0

        # builtin.split: (128x10x56x56xf32, 128x30x56x56xf32) <- ([128x10x56x56xf32, 128x30x56x56xf32])
        (
            split_1,
            split_2,
        ) = split_0
        del split_0

        # pd_op.conv2d: (128x10x56x56xf32) <- (128x10x56x56xf32, 10x10x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(
            split_1, parameter_108, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_108

        # builtin.combine: ([128x10x56x56xf32, 128x30x56x56xf32]) <- (128x10x56x56xf32, 128x30x56x56xf32)
        combine_0 = [conv2d_1, split_2]

        # pd_op.concat: (128x40x56x56xf32) <- ([128x10x56x56xf32, 128x30x56x56xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        del combine_0

        # pd_op.conv2d: (128x80x56x56xf32) <- (128x40x56x56xf32, 80x40x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(
            concat_0, parameter_107, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_107

        # pd_op.batch_norm_: (128x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x56x56xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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

        # pd_op.gelu: (128x80x56x56xf32) <- (128x80x56x56xf32)
        gelu_0 = paddle._C_ops.gelu(batch_norm__6, False)

        # pd_op.conv2d: (128x40x56x56xf32) <- (128x80x56x56xf32, 40x80x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(
            gelu_0, parameter_102, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_102

        # pd_op.add: (128x40x56x56xf32) <- (128x40x56x56xf32, 128x40x56x56xf32)
        add_1 = paddle._C_ops.add(batch_norm__0, conv2d_3)

        # pd_op.conv2d: (128x80x28x28xf32) <- (128x40x56x56xf32, 80x40x2x2xf32)
        conv2d_4 = paddle._C_ops.conv2d(
            add_1, parameter_101, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_101

        # pd_op.batch_norm_: (128x80x28x28xf32, 80xf32, 80xf32, 80xf32, 80xf32, -1xui8) <- (128x80x28x28xf32, 80xf32, 80xf32, 80xf32, 80xf32)
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
        del parameter_100, parameter_97, parameter_98, parameter_99

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [20, 60]

        # pd_op.split: ([128x20x28x28xf32, 128x60x28x28xf32]) <- (128x80x28x28xf32, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(batch_norm__12, full_int_array_1, full_0)

        # builtin.split: (128x20x28x28xf32, 128x60x28x28xf32) <- ([128x20x28x28xf32, 128x60x28x28xf32])
        (
            split_4,
            split_5,
        ) = split_3
        del split_3

        # pd_op.conv2d: (128x20x28x28xf32) <- (128x20x28x28xf32, 20x20x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(
            split_4, parameter_96, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_96

        # builtin.combine: ([128x20x28x28xf32, 128x60x28x28xf32]) <- (128x20x28x28xf32, 128x60x28x28xf32)
        combine_1 = [conv2d_5, split_5]

        # pd_op.concat: (128x80x28x28xf32) <- ([128x20x28x28xf32, 128x60x28x28xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)
        del combine_1

        # pd_op.conv2d: (128x160x28x28xf32) <- (128x80x28x28xf32, 160x80x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(
            concat_1, parameter_95, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_95

        # pd_op.batch_norm_: (128x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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
                parameter_94,
                parameter_93,
                parameter_92,
                parameter_91,
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
        del parameter_91, parameter_92, parameter_93, parameter_94

        # pd_op.gelu: (128x160x28x28xf32) <- (128x160x28x28xf32)
        gelu_1 = paddle._C_ops.gelu(batch_norm__18, False)

        # pd_op.conv2d: (128x80x28x28xf32) <- (128x160x28x28xf32, 80x160x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(
            gelu_1, parameter_90, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_90

        # pd_op.add: (128x80x28x28xf32) <- (128x80x28x28xf32, 128x80x28x28xf32)
        add_2 = paddle._C_ops.add(batch_norm__12, conv2d_7)

        # pd_op.split: ([128x20x28x28xf32, 128x60x28x28xf32]) <- (128x80x28x28xf32, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(add_2, full_int_array_1, full_0)
        del full_int_array_1

        # builtin.split: (128x20x28x28xf32, 128x60x28x28xf32) <- ([128x20x28x28xf32, 128x60x28x28xf32])
        (
            split_7,
            split_8,
        ) = split_6
        del split_6

        # pd_op.conv2d: (128x20x28x28xf32) <- (128x20x28x28xf32, 20x20x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(
            split_7, parameter_89, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_89

        # builtin.combine: ([128x20x28x28xf32, 128x60x28x28xf32]) <- (128x20x28x28xf32, 128x60x28x28xf32)
        combine_2 = [conv2d_8, split_8]

        # pd_op.concat: (128x80x28x28xf32) <- ([128x20x28x28xf32, 128x60x28x28xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        del combine_2

        # pd_op.conv2d: (128x160x28x28xf32) <- (128x80x28x28xf32, 160x80x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(
            concat_2, parameter_88, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_88

        # pd_op.batch_norm_: (128x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x28x28xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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

        # pd_op.gelu: (128x160x28x28xf32) <- (128x160x28x28xf32)
        gelu_2 = paddle._C_ops.gelu(batch_norm__24, False)

        # pd_op.conv2d: (128x80x28x28xf32) <- (128x160x28x28xf32, 80x160x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(
            gelu_2, parameter_83, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_83

        # pd_op.add: (128x80x28x28xf32) <- (128x80x28x28xf32, 128x80x28x28xf32)
        add_3 = paddle._C_ops.add(add_2, conv2d_10)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x80x28x28xf32, 160x80x2x2xf32)
        conv2d_11 = paddle._C_ops.conv2d(
            add_3, parameter_82, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_82

        # pd_op.batch_norm_: (128x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32, -1xui8) <- (128x160x14x14xf32, 160xf32, 160xf32, 160xf32, 160xf32)
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [40, 120]

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(batch_norm__30, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_10,
            split_11,
        ) = split_9
        del split_9

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(
            split_10, parameter_77, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_77

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_3 = [conv2d_12, split_11]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_0)
        del combine_3

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(
            concat_3, parameter_76, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_76

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_3 = paddle._C_ops.gelu(batch_norm__36, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(
            gelu_3, parameter_71, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_71

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_4 = paddle._C_ops.add(batch_norm__30, conv2d_14)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(add_4, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_13,
            split_14,
        ) = split_12
        del split_12

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(
            split_13, parameter_70, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_70

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_4 = [conv2d_15, split_14]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_0)
        del combine_4

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(
            concat_4, parameter_69, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_69

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_4 = paddle._C_ops.gelu(batch_norm__42, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(
            gelu_4, parameter_64, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_64

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_5 = paddle._C_ops.add(add_4, conv2d_17)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(add_5, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_16,
            split_17,
        ) = split_15
        del split_15

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(
            split_16, parameter_63, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_63

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_5 = [conv2d_18, split_17]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_0)
        del combine_5

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(
            concat_5, parameter_62, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_62

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_5 = paddle._C_ops.gelu(batch_norm__48, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(
            gelu_5, parameter_57, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_57

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_6 = paddle._C_ops.add(add_5, conv2d_20)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(add_6, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_19,
            split_20,
        ) = split_18
        del split_18

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(
            split_19, parameter_56, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_56

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_6 = [conv2d_21, split_20]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_0)
        del combine_6

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(
            concat_6, parameter_55, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_55

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_6 = paddle._C_ops.gelu(batch_norm__54, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(
            gelu_6, parameter_50, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_50

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_7 = paddle._C_ops.add(add_6, conv2d_23)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(add_7, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_22,
            split_23,
        ) = split_21
        del split_21

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(
            split_22, parameter_49, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_49

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_7 = [conv2d_24, split_23]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_0)
        del combine_7

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(
            concat_7, parameter_48, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_48

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_7 = paddle._C_ops.gelu(batch_norm__60, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(
            gelu_7, parameter_43, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_43

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_8 = paddle._C_ops.add(add_7, conv2d_26)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(add_8, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_25,
            split_26,
        ) = split_24
        del split_24

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(
            split_25, parameter_42, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_42

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_8 = [conv2d_27, split_26]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_0)
        del combine_8

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(
            concat_8, parameter_41, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_41

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_8 = paddle._C_ops.gelu(batch_norm__66, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(
            gelu_8, parameter_36, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_36

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_9 = paddle._C_ops.add(add_8, conv2d_29)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_27 = paddle._C_ops.split(add_9, full_int_array_2, full_0)

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_28,
            split_29,
        ) = split_27
        del split_27

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(
            split_28, parameter_35, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_35

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_9 = [conv2d_30, split_29]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_0)
        del combine_9

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(
            concat_9, parameter_34, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_34

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_9 = paddle._C_ops.gelu(batch_norm__72, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(
            gelu_9, parameter_29, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_29

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_10 = paddle._C_ops.add(add_9, conv2d_32)

        # pd_op.split: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x160x14x14xf32, 2xi64, 1xi32)
        split_30 = paddle._C_ops.split(add_10, full_int_array_2, full_0)
        del full_int_array_2

        # builtin.split: (128x40x14x14xf32, 128x120x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32])
        (
            split_31,
            split_32,
        ) = split_30
        del split_30

        # pd_op.conv2d: (128x40x14x14xf32) <- (128x40x14x14xf32, 40x40x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(
            split_31, parameter_28, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_28

        # builtin.combine: ([128x40x14x14xf32, 128x120x14x14xf32]) <- (128x40x14x14xf32, 128x120x14x14xf32)
        combine_10 = [conv2d_33, split_32]

        # pd_op.concat: (128x160x14x14xf32) <- ([128x40x14x14xf32, 128x120x14x14xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_0)
        del combine_10

        # pd_op.conv2d: (128x320x14x14xf32) <- (128x160x14x14xf32, 320x160x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(
            concat_10, parameter_27, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_27

        # pd_op.batch_norm_: (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x14x14xf32, 320xf32, 320xf32, 320xf32, 320xf32)
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

        # pd_op.gelu: (128x320x14x14xf32) <- (128x320x14x14xf32)
        gelu_10 = paddle._C_ops.gelu(batch_norm__78, False)

        # pd_op.conv2d: (128x160x14x14xf32) <- (128x320x14x14xf32, 160x320x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(
            gelu_10, parameter_22, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_22

        # pd_op.add: (128x160x14x14xf32) <- (128x160x14x14xf32, 128x160x14x14xf32)
        add_11 = paddle._C_ops.add(add_10, conv2d_35)

        # pd_op.conv2d: (128x320x7x7xf32) <- (128x160x14x14xf32, 320x160x2x2xf32)
        conv2d_36 = paddle._C_ops.conv2d(
            add_11, parameter_21, [2, 2], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_21

        # pd_op.batch_norm_: (128x320x7x7xf32, 320xf32, 320xf32, 320xf32, 320xf32, -1xui8) <- (128x320x7x7xf32, 320xf32, 320xf32, 320xf32, 320xf32)
        (
            batch_norm__84,
            batch_norm__85,
            batch_norm__86,
            batch_norm__87,
            batch_norm__88,
            batch_norm__89,
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

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [80, 240]

        # pd_op.split: ([128x80x7x7xf32, 128x240x7x7xf32]) <- (128x320x7x7xf32, 2xi64, 1xi32)
        split_33 = paddle._C_ops.split(batch_norm__84, full_int_array_3, full_0)

        # builtin.split: (128x80x7x7xf32, 128x240x7x7xf32) <- ([128x80x7x7xf32, 128x240x7x7xf32])
        (
            split_34,
            split_35,
        ) = split_33
        del split_33

        # pd_op.conv2d: (128x80x7x7xf32) <- (128x80x7x7xf32, 80x80x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(
            split_34, parameter_16, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_16

        # builtin.combine: ([128x80x7x7xf32, 128x240x7x7xf32]) <- (128x80x7x7xf32, 128x240x7x7xf32)
        combine_11 = [conv2d_37, split_35]

        # pd_op.concat: (128x320x7x7xf32) <- ([128x80x7x7xf32, 128x240x7x7xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_0)
        del combine_11

        # pd_op.conv2d: (128x640x7x7xf32) <- (128x320x7x7xf32, 640x320x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(
            concat_11, parameter_15, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_15

        # pd_op.batch_norm_: (128x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (128x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        (
            batch_norm__90,
            batch_norm__91,
            batch_norm__92,
            batch_norm__93,
            batch_norm__94,
            batch_norm__95,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_38,
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

        # pd_op.gelu: (128x640x7x7xf32) <- (128x640x7x7xf32)
        gelu_11 = paddle._C_ops.gelu(batch_norm__90, False)

        # pd_op.conv2d: (128x320x7x7xf32) <- (128x640x7x7xf32, 320x640x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(
            gelu_11, parameter_10, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_10

        # pd_op.add: (128x320x7x7xf32) <- (128x320x7x7xf32, 128x320x7x7xf32)
        add_12 = paddle._C_ops.add(batch_norm__84, conv2d_39)

        # pd_op.split: ([128x80x7x7xf32, 128x240x7x7xf32]) <- (128x320x7x7xf32, 2xi64, 1xi32)
        split_36 = paddle._C_ops.split(add_12, full_int_array_3, full_0)
        del full_int_array_3

        # builtin.split: (128x80x7x7xf32, 128x240x7x7xf32) <- ([128x80x7x7xf32, 128x240x7x7xf32])
        (
            split_37,
            split_38,
        ) = split_36
        del split_36

        # pd_op.conv2d: (128x80x7x7xf32) <- (128x80x7x7xf32, 80x80x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(
            split_37, parameter_9, [1, 1], [1, 1], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_9

        # builtin.combine: ([128x80x7x7xf32, 128x240x7x7xf32]) <- (128x80x7x7xf32, 128x240x7x7xf32)
        combine_12 = [conv2d_40, split_38]

        # pd_op.concat: (128x320x7x7xf32) <- ([128x80x7x7xf32, 128x240x7x7xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_0)
        del combine_12

        # pd_op.conv2d: (128x640x7x7xf32) <- (128x320x7x7xf32, 640x320x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(
            concat_12, parameter_8, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_8

        # pd_op.batch_norm_: (128x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32, -1xui8) <- (128x640x7x7xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        (
            batch_norm__96,
            batch_norm__97,
            batch_norm__98,
            batch_norm__99,
            batch_norm__100,
            batch_norm__101,
        ) = (lambda x, f: f(x))(
            paddle._C_ops.batch_norm(
                conv2d_41,
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

        # pd_op.gelu: (128x640x7x7xf32) <- (128x640x7x7xf32)
        gelu_12 = paddle._C_ops.gelu(batch_norm__96, False)

        # pd_op.conv2d: (128x320x7x7xf32) <- (128x640x7x7xf32, 320x640x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(
            gelu_12, parameter_3, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_3

        # pd_op.add: (128x320x7x7xf32) <- (128x320x7x7xf32, 128x320x7x7xf32)
        add_13 = paddle._C_ops.add(add_12, conv2d_42)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (128x320x1x1xf32) <- (128x320x7x7xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            add_13,
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

        # pd_op.conv2d: (128x1280x1x1xf32) <- (128x320x1x1xf32, 1280x320x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(
            pool2d_0, parameter_2, [1, 1], [0, 0], "EXPLICIT", [1, 1], 1, "NCHW"
        )
        del parameter_2

        # pd_op.gelu: (128x1280x1x1xf32) <- (128x1280x1x1xf32)
        gelu_13 = paddle._C_ops.gelu(conv2d_43, False)

        # pd_op.flatten: (128x1280xf32) <- (128x1280x1x1xf32)
        flatten_0 = paddle._C_ops.flatten(gelu_13, 1, 3)

        # pd_op.matmul: (128x102xf32) <- (128x1280xf32, 1280x102xf32)
        matmul_0 = paddle._C_ops.matmul(flatten_0, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (128x102xf32) <- (128x102xf32, 102xf32)
        add_0 = paddle._C_ops.add(matmul_0, parameter_0)
        del (
            add_1,
            add_10,
            add_11,
            add_12,
            add_13,
            add_2,
            add_3,
            add_4,
            add_5,
            add_6,
            add_7,
            add_8,
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
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            batch_norm__0,
            batch_norm__1,
            batch_norm__10,
            batch_norm__100,
            batch_norm__101,
            batch_norm__11,
            batch_norm__12,
            batch_norm__13,
            batch_norm__14,
            batch_norm__15,
            batch_norm__16,
            batch_norm__17,
            batch_norm__18,
            batch_norm__19,
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
            concat_10,
            concat_11,
            concat_12,
            concat_2,
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
            conv2d_5,
            conv2d_6,
            conv2d_7,
            conv2d_8,
            conv2d_9,
            flatten_0,
            full_0,
            full_int_array_4,
            gelu_0,
            gelu_1,
            gelu_10,
            gelu_11,
            gelu_12,
            gelu_13,
            gelu_2,
            gelu_3,
            gelu_4,
            gelu_5,
            gelu_6,
            gelu_7,
            gelu_8,
            gelu_9,
            matmul_0,
            parameter_0,
            pool2d_0,
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
            split_5,
            split_7,
            split_8,
        )

        return add_0
